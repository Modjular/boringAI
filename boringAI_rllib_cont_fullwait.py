# Rllib docs: https://docs.ray.io/en/latest/rllib.html
# 
# This AI waits for the block to break before moving on to the next step


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
from gym.spaces import Discrete, MultiDiscrete
from ray.rllib.agents import ppo
import random
import datetime

INITIAL_REWARD = 400

class BoringAI(gym.Env):

    def __init__(self, env_config):  
        # Static Parameters
        self.size = 10
        self.reward_density = .1
        self.penalty_density = .02
        self.obs_size = 1
        self.max_episode_steps = 100
        self.log_frequency = 5
        self.start_time = datetime.datetime.now()
        print("Init at ", self.start_time)

        # BoringAI parameters
        self.tunnel_len = 9
        self.block_type = ['dirt', 'stone', 'planks']
        self.block_dict = {'air': 0, 'dirt': 1, 'stone': 2, 'planks': 3}

        # Rllib Parameters
        # self.action_space = Box(-1, 1, shape=(3,), dtype=np.float32)
        # self.observation_space = Box(0, 1, shape=(np.prod([2, self.obs_size, self.obs_size]), ), dtype=np.int32)
        self.action_space = Discrete(3)                 # [pickaxe, shovel, axe]
        self.observation_space = MultiDiscrete([4,2]);  # ([air, dirt, stone, planks], [no attack, attack])

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
        self.allow_break_action = False
        self.episode_step = 0
        self.episode_return = 0
        self.returns = []
        self.steps = []
        self.initial_reward = INITIAL_REWARD  # used per episode to keep track of decreasing reward

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
        current_step = self.steps[-1] if len(self.steps) > 0 else 0
        self.steps.append(current_step + self.episode_step)
        self.episode_return = 0
        self.episode_step = 0
        self.initial_reward = INITIAL_REWARD


        # Log
        if len(self.returns) > self.log_frequency and \
            len(self.returns) % self.log_frequency == 0:
            self.log_returns()

        # Get Observation
        self.obs, self.allow_break_action = self.get_observation(world_state)

        # return self.obs.flatten()
        return [self.obs, 1 if self.allow_break_action else 0]

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
        reward_delta = 0 # calling getWorldState() below loses our reward delta 

        # Get Action
        if self.allow_break_action:     
            
            # switch tools, start digging
            self.agent_host.sendCommand('move 0');
            self.agent_host.sendCommand('hotbar.' + str(action + 1) + ' 1')
            self.agent_host.sendCommand('attack 1')
            # time.sleep(0.8)  # Allow steve to break the block

            # TODO
            # Add a while loop that lets us finish destroying the block
            # Or maybe this is a behavior it can learn on its own?
            # Otherwise, it might switch randomly, resetting the breaking

            while self.allow_break_action:
                time.sleep(0.1) # give it a sec to do its thang
                world_state = self.agent_host.getWorldState()
                for r in world_state.rewards:   # these deltas will be lost otherwise
                    reward_delta += r.getValue()
                _, self.allow_break_action = self.get_observation(world_state)

        else:                           
            
            # just move forward if no block in front
            self.agent_host.sendCommand('attack 0')
            self.agent_host.sendCommand('move 1');
            time.sleep(0.2)

        self.episode_step += 1


        # Get Observation
        world_state = self.agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:", error.text)
        self.obs, self.allow_break_action = self.get_observation(world_state) 

        # Get Done
        done = not world_state.is_mission_running 

        # Get Reward
        reward = 0
        for r in world_state.rewards:
            reward_delta += r.getValue()
        self.initial_reward += reward_delta     # decrement reward
        print("REWARD DELTA: ", reward_delta)
        
        # NOTE
        # "Error: AgentHost::sendCommand : commands connection is not open. Is the mission running?"
        # The error above causes <RewardForTimeTaken> to register inconsistently
        # To be consistent, we keep track of self.initial_reward internally
        
        if done:
            reward += self.initial_reward   # add to reward so it can get passed out
            print("END REWARD", self.initial_reward)

        self.episode_return += reward

        return [self.obs, 1 if self.allow_break_action else 0], reward, done, dict()

    def get_mission_xml(self):
        
        tunnel_xml = ''
        
        # Finish Line
        for i in range(-5, 6):
            if i%2 == 0:
                tunnel_xml += "<DrawBlock x='{}' y='1' z='{}' type='coal_block'   />".format(i, self.tunnel_len + 1)
            else:
                tunnel_xml += "<DrawBlock x='{}' y='1' z='{}' type='quartz_block' />".format(i, self.tunnel_len + 1)

        # Glass Box
        tunnel_xml += "<DrawCuboid x1='-5' x2='5'  y1='2' y2='4' z1='1' z2='1'  type='glass'/>"
        tunnel_xml += "<DrawCuboid x1='-5' x2='-5' y1='2' y2='4' z1='1' z2='{}' type='glass'/>".format(self.tunnel_len)
        tunnel_xml += "<DrawCuboid x1='5'  x2='5'  y1='2' y2='4' z1='1' z2='{}' type='glass'/>".format(self.tunnel_len)

        # Draw entrance
        #tunnel_xml += "<DrawBlock x='0' y='2' z='1' type='air' />"
        tunnel_xml += "<DrawBlock x='0' y='3' z='1' type='air' />"

        # NOTE
        # Was running into wild results because lots of stone = bad results
        # while lots of dirt = quick results
        # But it wasn't really learning. Just chance
        # Defining equal amounts of each block = more consistency
        blocks = ['stone', 'stone', 'stone', 'dirt', 'dirt', 'dirt', 'planks', 'planks', 'planks']
        random.shuffle(blocks)  

        # Blocks to Break
        for i in range(0, self.tunnel_len):
            tunnel_xml += "<DrawBlock x='0' y='2' z='{}' type='{}' />".format(i + 1, blocks[i])
            # tunnel_xml += "<DrawBlock x=\'0\' y=\'3\' z=\'" + str(i) + "\' type=\'" + random_block + "\' />"      # uncomment for 2nd row

 
        return '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
                <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">

                    <About>
                        <Summary>Tunnel Crawler</Summary>
                    </About>

                    <ServerSection>
                        <ServerInitialConditions>
                            <Time>
                                <StartTime>0</StartTime>
                                <AllowPassageOfTime>true</AllowPassageOfTime>
                            </Time>
                            <Weather>clear</Weather>
                        </ServerInitialConditions>
                        <ServerHandlers>
                            <FlatWorldGenerator generatorString="3;7,2;1;"/>
                            <DrawingDecorator>''' + \
                                "<DrawCuboid x1='{}' x2='{}' y1='2' y2='2' z1='{}' z2='{}' type='air'/>".format(-self.size, self.size, -self.size, self.size) + \
                                "<DrawCuboid x1='{}' x2='{}' y1='1' y2='1' z1='{}' z2='{}' type='bedrock'/>".format(-self.size, self.size, -self.size, self.size) + \
                                tunnel_xml + \
                                '''
                                <DrawBlock x='0'  y='2' z='0' type='air' />
                                <DrawBlock x='0'  y='3' z='0' type='air' />
                            </DrawingDecorator>
                            <ServerQuitWhenAnyAgentFinishes/>
                        </ServerHandlers>
                    </ServerSection>

                    <AgentSection mode="Survival">
                        <Name>BoringAI</Name>
                        <AgentStart>
                            <Placement x="0.5" y="2" z="0.5" pitch="64" yaw="0"/>
                            <Inventory>
                                <InventoryItem slot="0" type="diamond_pickaxe"/>
                                <InventoryItem slot="1" type="diamond_shovel"/>
                                <InventoryItem slot="2" type="diamond_axe"/>
                            </Inventory>
                        </AgentStart>
                        <AgentHandlers>
                            <ContinuousMovementCommands/>
                            <InventoryCommands/>
                            <ObservationFromFullInventory flat="false"/>
                            <ObservationFromFullStats/>
                            <ObservationFromRay/>
                            <ObservationFromGrid>
                                <Grid name="floorAll">
                                <min x="0" y="0" z="1"/>
                                <max x="0" y="0" z="'''+str(int(self.obs_size))+'''"/>
                                </Grid>
                            </ObservationFromGrid>
                            <RewardForTimeTaken initialReward="1000"  delta="-1" density= "PER_TICK"/>
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
        Use the agent observation API to get the block right in front of it (text)

        Args
            world_state: <object> current agent world state

        Returns
            observation: <np.array>
        """
        obs = 0
        allow_break_action = False

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
                obs = observations['floorAll']
                obs = self.block_dict[obs[0]];

                # For extra safety, the ground is unbreakable bedrock
                allow_break_action = observations['LineOfSight']['type'] != 'bedrock'
                
                break

        return obs, allow_break_action

    def log_returns(self):
        """
        Log the current returns as a graph and text file

        Args:
            steps (list): list of global steps after each episode
            returns (list): list of total return of each episode
        """
        box = np.ones(self.log_frequency) / self.log_frequency
        returns_smooth = np.convolve(self.returns, box, mode='same')
        plt.clf()
        plt.plot(self.steps, returns_smooth)
        plt.title('BoringAI Continuous Sparse')
        plt.ylabel('Return')
        plt.xlabel('Steps')
        plt.savefig('returns.png')

        with open('returns.txt', 'w') as f:
            for step, value in zip(self.steps, self.returns):
                f.write("{}\t{}\n".format(step, value)) 


if __name__ == '__main__':
    ray.init()
    trainer = ppo.PPOTrainer(env=BoringAI, config={
        'env_config': {},           # No environment parameters to configure
        'framework': 'torch',       # Use pyotrch instead of tensorflow
        'num_gpus': 0,              # We aren't using GPUs
        'num_workers': 0            # We aren't using parallelism
    })

    while True:
        print(trainer.train())