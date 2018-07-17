from pysc2.agents import base_agent
from pysc2.lib import actions, features
from pysc2.env import sc2_env
from absl import app

import time


# Constants definitions
# Functions
_BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_NOOP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_MOVE_CAMERA = actions.FUNCTIONS.move_camera.id
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_RALLY_UNITS_MINIMAP = actions.FUNCTIONS.Rally_Units_minimap.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id

# Features
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index

# Unit IDs
_TERRAN_COMMAND_CENTER = 18
_TERRAN_SUPPLY_DEPOT = 19
_TERRAN_SCV = 45
_TERRAN_BARRACKS = 21

# Parameters
_PLAYER_SELF = 1
_NOT_QUEUED = [0]
_QUEUED = [1]
_SUPPLY_USED = 3
_SUPPLY_MAX = 4


class ScriptedAgent(base_agent.BaseAgent):
    base_top_left = None
    supply_depot_built = False
    scv_selected = False
    barracks_built = False
    barracks_selected = False
    barracks_rallied = False
    army_selected = False
    army_rallied = False

    def transformLocation(self, x, x_distance, y, y_distance):
        if not self.base_top_left:
            return [x - x_distance, y- y_distance]

        return [x + x_distance, y + y_distance]

    def step(self, obs):
        super(ScriptedAgent, self).step(obs)

        time.sleep(0.5)

        if self.base_top_left is None:
            # print(obs.observation)
            y, x = (obs.observation["feature_minimap"][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
            target = [int(x.mean()), int(y.mean())]
            self.base_top_left = y.mean()<=31
            return actions.FunctionCall(_MOVE_CAMERA, [target])

        if not self.supply_depot_built:
            if not self.scv_selected:
                unit_type = obs.observation["feature_screen"][_UNIT_TYPE]

                unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero()

                if (unit_y is not []) and (unit_x is not []):
                    target = [unit_x[0], unit_y[0]]
                    self.scv_selected = True

                return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
            elif _BUILD_SUPPLY_DEPOT in obs.observation["available_actions"]:
                unit_type = obs.observation["feature_screen"][_UNIT_TYPE]
                unit_y, unit_x = (unit_type == _TERRAN_COMMAND_CENTER).nonzero()
                target = self.transformLocation(int(unit_x.mean()), 0, int(unit_y.mean()), 20)
                self.supply_depot_built = True

                return actions.FunctionCall(_BUILD_SUPPLY_DEPOT, [_NOT_QUEUED, target])

        elif not self.barracks_built:
            if _BUILD_BARRACKS in obs.observation["available_actions"]:
                unit_type = obs.observation["feature_screen"][_UNIT_TYPE]
                unit_y, unit_x = (unit_type == _TERRAN_COMMAND_CENTER).nonzero()
                target = self.transformLocation(int(unit_x.mean()), 20, int(unit_y.mean()), 0)
                self.barracks_built = True

                return actions.FunctionCall(_BUILD_BARRACKS, [_NOT_QUEUED, target])

        elif not self.barracks_rallied:
            if not self.barracks_selected:
                unit_type = obs.observation["feature_screen"][_UNIT_TYPE]
                unit_y, unit_x = (unit_type == _TERRAN_BARRACKS).nonzero()

                if unit_y.any():
                    target = [int(unit_x.mean()), int(unit_y.mean())]

                    self.barracks_selected = True

                    return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
            else:
                self.barracks_rallied = True

                if self.base_top_left:
                    return actions.FunctionCall(_RALLY_UNITS_MINIMAP, [_NOT_QUEUED, [29, 21]])

                return actions.FunctionCall(_RALLY_UNITS_MINIMAP, [_NOT_QUEUED, [29, 46]])

        elif obs.observation["player"][_SUPPLY_USED] < obs.observation["player"][_SUPPLY_MAX] \
            and _TRAIN_MARINE in obs.observation["available_actions"]:
            return actions.FunctionCall(_TRAIN_MARINE, [_QUEUED])

        elif not self.army_rallied:
            if not self.army_selected:
                if _SELECT_ARMY in obs.observation["available_actions"]:
                    self.army_selected = True
                    self.barracks_selected = False

                    return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])

            elif _ATTACK_MINIMAP in obs.observation["available_actions"]:
                self.army_rallied = True
                self.army_selected = False

                if self.base_top_left:
                    return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, [39, 45]])

                return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, [21, 24]])


        return actions.FUNCTIONS.no_op()


def main(unused_argv):
    agent = ScriptedAgent()
    try:
        while True:
            with sc2_env.SC2Env(
                map_name="Simple64",
                players=[
                    sc2_env.Agent(sc2_env.Race.terran),
                    sc2_env.Bot(sc2_env.Race.random, sc2_env.Difficulty.very_easy)
                ],
                agent_interface_format=features.AgentInterfaceFormat(
                    feature_dimensions=features.Dimensions(screen=84, minimap=64)),
                step_mul=16,
                game_steps_per_episode=0,
                visualize=True
                ) as env:
                agent.setup(env.observation_spec(), env.action_spec())

                timesteps = env.reset()
                agent.reset()

                while True:
                    step_actions = [agent.step(timesteps[0])]
                    if timesteps[0].last():
                        break;
                    timesteps = env.step(step_actions)

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    app.run(main)