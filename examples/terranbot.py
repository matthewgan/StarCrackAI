from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app
import random


class TerranAgent(base_agent.BaseAgent):
    def __init__(self):
        super(TerranAgent, self).__init__()
        self.attack_coordinates = None
        self.supply_depot_building = False
        self.scv_selected = False
        self.barracks_building = False
        self.barracks_selected = False
        self.barracks_rallied = False
        self.army_selected = False
        self.army_rallied = False
        self.scv_building = False
        self.marine_building = False

    def unit_type_is_selected(self, obs, unit_type):
        if (len(obs.observation.single_select) > 0 and
                obs.observation.single_select[0].unit_type == unit_type):
            return True

        if (len(obs.observation.multi_select) > 0 and
                obs.observation.multi_select[0].unit_type == unit_type):
            return True
        return False

    def get_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.feature_units
                if unit.unit_type == unit_type]

    def can_do(self, obs, action):
        return action in obs.observation.available_actions

    def step(self, obs):
        super(TerranAgent, self).step()

        if obs.first():
            player_y, player_x = (obs.observation.feature_minimap.player_relative ==
                                  features.PlayerRelative.SELF).nonzero()
            xmean = player_x.mean()
            ymean = player_y.mean()

            if xmean <= 31 and ymean <= 31:
                self.attack_coordinates = (49, 49)
            else:
                self.attack_coordinates = (12, 16)

        cmdcenters = self.get_units_by_type(obs, units.Terran.CommandCenter)
        scvs = self.get_units_by_type(obs, units.Terran.SCV)
        barracks = self.get_units_by_type(obs, units.Terran.Barracks)
        depots = self.get_units_by_type(obs, units.Terran.SupplyDepot)
        marines = self.get_units_by_type(obs, units.Terran.Marine)

        supply_cap = obs.observation.player.food_cap
        supply_used = obs.observation.player.food_used
        supply_free = supply_cap - supply_used
        supply_army = obs.observation.player.food_army
        supply_worker = obs.observation.player.food_workers
        minerals = obs.observation.player.minerals

        if self.supply_depot_building==False and self.barracks_building==False and self.scv_building==False:
            if supply_free <= 0 and self.can_do(obs, actions.FUNCTIONS.Build_SupplyDepot_screen.id):
                # Build supply depot
                self.supply_depot_building = True
            elif len(barracks)<5 and self.can_do(obs, actions.FUNCTIONS.Build_Barracks_screen.id):
                self.barracks_building = True
            elif len(scvs)<24 and self.can_do(obs, actions.FUNCTIONS.Train_SCV_quick.id):
                self.scv_building = True
            elif self.can_do(obs, actions.FUNCTIONS.Train_Marine_quick.id):
                self.marine_building = True

        if self.supply_depot_building:
            if self.unit_type_is_selected(obs, units.Terran.SCV):
                x = random.randint(0, 83)
                y = random.randint(0, 83)
                self.supply_depot_building = False
                return actions.FUNCTIONS.Build_SupplyDepot_screen("now", (x, y))
            else:
                scv = random.choice(scvs)
                return actions.FUNCTIONS.select_point("select all type", (scv.x, scv.y))

        if self.barracks_building:
            if self.unit_type_is_selected(obs, units.Terran.SCV):
                x = random.randint(0, 83)
                y = random.randint(0, 83)
                self.barracks_building = False
                return actions.FUNCTIONS.Build_Barracks_screen("now", (x, y))
            else:
                scv = random.choice(scvs)
                return actions.FUNCTIONS.select_point("select all type", (scv.x, scv.y))

        if self.scv_building:
            if self.unit_type_is_selected(obs, units.Terran.CommandCenter):
                self.scv_building = False
                return actions.FUNCTIONS.Train_SCV_quick("now")
            else:
                cc = random.choice(cmdcenters)
                return actions.FUNCTIONS.select_point("select all type", (cc.x, cc.y))

        if self.marine_building:
            if self.unit_type_is_selected(obs, units.Terran.Barracks):
                self.marine_building = False
                return actions.FUNCTIONS.Train_Marine_quick("now")
            else:
                barrack = random.choice(barracks)
                return actions.FUNCTIONS.select_point("select all type", (barrack.x, barrack.y))

        return actions.FUNCTIONS.no_op()
