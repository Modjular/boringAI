# Rllib docs: https://docs.ray.io/en/latest/rllib.html

try:
    from malmo import MalmoPython
except:
    import MalmoPython

import sys
import time
import json
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randint

import gym, ray
from gym.spaces import Discrete, Box
from ray.rllib.agents import ppo


class DiamondCollector(gym.Env):

    def __init__(self, env_config):  
        # Static Parameters
        self.size = 50
        self.reward_density = .1
        self.penalty_density = .02
        self.obs_size = 5
        self.max_episode_steps = 100
        self.log_frequency = 10
        self.action_dict = {
            0: 'move 1',  # Move one block forward
            1: 'turn 1',  # Turn 90 degrees to the right
            2: 'turn -1',  # Turn 90 degrees to the left
            3: 'attack 1'  # Destroy block
        }

        # Rllib Parameters
        self.action_space = Discrete(len(self.action_dict))
        self.observation_space = Box(0, 1, shape=(np.prod([2, self.obs_size, self.obs_size]), ), dtype=np.int32)

        # Malmo Parameters
        self.agent_host = MalmoPython.AgentHost()
        try:
            self.agent_host.parse( sys.argv )
        except RuntimeError as e:
            print('ERROR:', e)
            print(self.agent_host.getUsage())
            exit(1)

        # DiamondCollector Parameters
        self.obs = None
        self.episode_step = 0
        self.episode_return = 0
        self.returns = []
        self.episode_num = 0
        self.episodes = []
        self.steps = []

    def reset(self):
        """
        Resets the environment for the next episode.

        Returns
            observation: <np.array> flattened initial obseravtion
        """
        # Reset Malmo
        world_state = self.init_malmo()

        # Reset Variables
        self.returns.append(self.episode_return)
        self.episodes.append(self.episode_num)
        self.episode_num += 1

        current_step = self.steps[-1] if len(self.steps) > 0 else 0
        self.steps.append(current_step + self.episode_step)
        self.episode_return = 0
        self.episode_step = 0

        # Log
        if len(self.returns) > 0:
            self.log_returns()

        # Get Observation
        self.obs = self.get_observation(world_state)

        return self.obs.flatten()

    def step(self, action):
        """
        Take an action in the environment and return the results.

        Args
            action: <int> index of the action to take

        Returns
            observation: <np.array> flattened array of obseravtion
            reward: <int> reward from taking action
            done: <bool> indicates terminal state
            info: <dict> dictionary of extra information
        """

        # Get Action
        command = self.action_dict[action]
        allow_break_action = self.obs[1, int(self.obs_size/2)-1, int(self.obs_size/2)] == 1
        if command != 'attack 1' or allow_break_action:
            self.agent_host.sendCommand(command)
            time.sleep(.1)
            self.episode_step += 1

        # Get Done
        done = False
        if self.episode_step >= self.max_episode_steps or \
                (self.obs[0, int(self.obs_size/2)-1, int(self.obs_size/2)] == 1 and \
                self.obs[1, int(self.obs_size/2)-1, int(self.obs_size/2)] == 0 and \
                command == 'move 1'):
            done = True
            time.sleep(2)  

        # Get Observation
        world_state = self.agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:", error.text)
        self.obs = self.get_observation(world_state) 

        # Get Reward
        reward = 0
        for r in world_state.rewards:
            reward += r.getValue()
        self.episode_return += reward

        return self.obs.flatten(), reward, done, dict()

    def get_mission_xml(self):
        block_type = ['dirt', 'stone']
        tunnel_xml = ''
        for i in range(1, self.tunnel_len + 1):
            tunnel_xml += "<DrawBlock x=\'0\' y=\'2\' z=\'" + str(i) + "\' type=\'" + random.choice(block_type) + "\' />"
        for i in range(-5, 6):
            if i%2 == 0:
                tunnel_xml += "<DrawBlock x=\'" + str(i) + "\' y=\'1\' z=\'" + str(self.tunnel_len) + "\' type=\'coal_block\' />"
            else:
                tunnel_xml += "<DrawBlock x=\'" + str(i) + "\' y=\'1\' z=\'" + str(self.tunnel_len) + "\' type=\'quartz_block\' />"
        for i in range(-5, 6):
            for j in range(2,5):
                tunnel_xml += "<DrawBlock x=\'" + str(i) + "\' y=\'" + str(j) + "\' z=\'1\' type=\'glass\' />"
        for i in range(1, self.tunnel_len + 1):
            for j in range(2, 5):
                tunnel_xml += "<DrawBlock x=\'-5\' y=\'" + str(j) + "\' z=\'"+ str(i) + "\' type=\'glass\' />"
                tunnel_xml += "<DrawBlock x=\'5\' y=\'" + str(j) + "\' z=\'"+ str(i) + "\' type=\'glass\' />"


        tunnel_xml += "<DrawBlock x=\'0\' y=\'2\' z=\'1\' type=\'air\' />"
        tunnel_xml += "<DrawBlock x=\'0\' y=\'3\' z=\'1\' type=\'air\' />"

        return '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
                <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">

                    <About>
                        <Summary>Tunnel Crawler</Summary>
                    </About>

                    <ServerSection>
                        <ServerInitialConditions>
                            <Time>
                                <StartTime>12000</StartTime>
                                <AllowPassageOfTime>true</AllowPassageOfTime>
                            </Time>
                            <Weather>clear</Weather>
                        </ServerInitialConditions>
                        <ServerHandlers>
                            <FlatWorldGenerator generatorString="3;7,2;1;"/>
                            <DrawingDecorator>''' + \
                                "<DrawCuboid x1='{}' x2='{}' y1='2' y2='2' z1='{}' z2='{}' type='air'/>".format(-self.size, self.size, -self.size, self.size) + \
                                "<DrawCuboid x1='{}' x2='{}' y1='1' y2='1' z1='{}' z2='{}' type='grass'/>".format(-self.size, self.size, -self.size, self.size) + \
                                tunnel_xml + \
                                '''<DrawBlock x='0'  y='2' z='0' type='air' />
                                <DrawBlock x='0'  y='1' z='0' type='grass' />
                            </DrawingDecorator>
                            <ServerQuitWhenAnyAgentFinishes/>
                        </ServerHandlers>
                    </ServerSection>

                    <AgentSection mode="Survival">
                        <Name>Tunnel Crawler</Name>
                        <AgentStart>
                            <Placement x="0.5" y="2" z="0.5" pitch="45" yaw="0"/>
                            <Inventory>
                                <InventoryItem slot="0" type="diamond_pickaxe"/>
                                <InventoryItem slot="1" type="diamond_shovel"/>
                            </Inventory>
                        </AgentStart>
                        <AgentHandlers>
                            <ContinuousMovementCommands/>
                            <InventoryCommands/>
                            <ObservationFromFullInventory flat="false"/>
                            <ObservationFromFullStats/>
                            <RewardForCollectingItem>
                                <Item reward='1' type='dirt'/>
                                <Item reward='1' type='stone'/>
                            </RewardForCollectingItem>
                            <ObservationFromGrid>
                                <Grid name="floorAll">
                                    <min x="-'''+str(int(self.obs_size/2))+'''" y="-1" z="-'''+str(int(self.obs_size/2))+'''"/>
                                    <max x="'''+str(int(self.obs_size/2))+'''" y="0" z="'''+str(int(self.obs_size/2))+'''"/>
                                </Grid>
                            </ObservationFromGrid>
                            <AgentQuitFromTouchingBlockType>
                                <Block type="coal_block"/>
                            </AgentQuitFromTouchingBlockType>
                        </AgentHandlers>
                    </AgentSection>
                </Mission>'''

    def init_malmo(self):
        """
        Initialize new malmo mission.
        """
        my_mission = MalmoPython.MissionSpec(self.get_mission_xml(), True)
        my_mission_record = MalmoPython.MissionRecordSpec()
        my_mission.requestVideo(800, 500)
        my_mission.setViewpoint(1)

        max_retries = 3
        my_clients = MalmoPython.ClientPool()
        my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000)) # add Minecraft machines here as available

        for retry in range(max_retries):
            try:
                self.agent_host.startMission( my_mission, my_clients, my_mission_record, 0, 'DiamondCollector' )
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    print("Error starting mission:", e)
                    exit(1)
                else:
                    time.sleep(2)

        world_state = self.agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()
            for error in world_state.errors:
                print("\nError:", error.text)

        return world_state

    def get_observation(self, world_state):
        """
        Use the agent observation API to get a 2 x 5 x 5 grid around the agent. 
        The agent is in the center square facing up.

        Args
            world_state: <object> current agent world state

        Returns
            observation: <np.array>
        """
        obs = np.zeros((2, self.obs_size, self.obs_size))

        while world_state.is_mission_running:
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()
            if len(world_state.errors) > 0:
                raise AssertionError('Could not load grid.')

            if world_state.number_of_observations_since_last_state > 0:
                # First we get the json from the observation API
                msg = world_state.observations[-1].text
                observations = json.loads(msg)

                # Get observation
                grid = observations['floorAll']
                grid_binary = [1 if x == 'diamond_ore' or x == 'lava' else 0 for x in grid]
                obs = np.reshape(grid_binary, (2, self.obs_size, self.obs_size))

                # Rotate observation with orientation of agent
                yaw = observations['Yaw']
                if yaw == 270:
                    obs = np.rot90(obs, k=1, axes=(1, 2))
                elif yaw == 0:
                    obs = np.rot90(obs, k=2, axes=(1, 2))
                elif yaw == 90:
                    obs = np.rot90(obs, k=3, axes=(1, 2))
                
                break

        return obs

    def log_returns(self):
        # box = np.ones(self.log_frequency) / self.log_frequency
        # returns_smooth = np.convolve(self.returns, box, mode='same')
        plt.clf()
        plt.plot(self.episodes[1:], self.returns[1:])
        plt.title('Diamond Collector')
        plt.ylabel('Return')
        plt.xlabel('Episode')
        plt.savefig('returns.png')

if __name__ == '__main__':
    ray.init()
    trainer = ppo.PPOTrainer(env=DiamondCollector, config={
        'env_config': {},           # No environment parameters to configure
        'framework': 'torch',       # Use pyotrch instead of tensorflow
        'num_gpus': 0,              # We aren't using GPUs
        'num_workers': 0            # We aren't using parallelism
    })

    while True:
        print(trainer.train())
