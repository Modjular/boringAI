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
from collections import defaultdict

import gym, ray
from gym.spaces import Discrete, MultiDiscrete
from ray.rllib.agents import ppo
import random
import datetime
import pandas as pd

INITIAL_REWARD = 0

class BoringAI(gym.Env):

    def __init__(self, env_config):
        # Static Parameters
        self.size = 10
        self.reward_density = .1
        self.penalty_density = .02
        self.obs_size = 1
        self.max_episode_steps = 100
        self.log_frequency = 5
        self.golden_durability = 2
        self.start_time = datetime.datetime.now()
        print("Init at ", self.start_time)

        # BoringAI parameters
        self.tunnel_len = 9
        self.block_type = ['stone', 'dirt', 'planks','prismarine','clay','log']
        self.block_dict = {'stone': 0, 'dirt': 1, 'planks': 2,
                'prismarine' :3, 'clay': 4, 'log':5,
                'bedrock':6, 'air':6}
        self.tool_dict = {0:"diamond pickaxe", 1:"diamond shovel",2:"diamond axe",
                3:"golden pickaxe", 4:"golden shovel", 5:"golden axe"}
        self.total_tool_type = 3
        self.total_tool_material = 2
        self.total_block_type = len(self.block_dict.keys()) - 1 #minus 1 due to air and bedrock counting as other
        self.golden_durability_dict = defaultdict(lambda:self.golden_durability) # currently only used for golden, takes in index of action
        # Rllib Parameters
        # self.action_space = Box(-1, 1, shape=(3,), dtype=np.float32)
        # self.observation_space = Box(0, 1, shape=(np.prod([2, self.obs_size, self.obs_size]), ), dtype=np.int32)
        self.action_space = MultiDiscrete([self.total_tool_type,self.total_tool_material]   )
        # [pickaxe, shovel, axe] [diamond,golden]
        self.observation_space = MultiDiscrete([self.total_block_type,2]);  # ([air, dirt, stone, planks], [no attack, attack])

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
        self.initial_reward = INITIAL_REWARD  # used per episode to keep track of ticks
        self.episode_action_log = defaultdict(lambda:[0,0]) # for each ep, [right uses,total]
        self.action_log = defaultdict(lambda:[]) #for each tool, has a list of percentage used correctly per episode

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
        self.golden_durability_dict.clear()
        self.golden_durablity_dict = defaultdict(lambda:self.golden_durability)
        #self.episode_action_log = defaultdict(lambda:[0,0])#uncomment when you want to have per episode statistics

        # Log
        if len(self.returns) > 0:
            self.log_returns()
        if len(self.action_log) >0:
            self.log_actions()

        # Get Observation
        self.obs, self.allow_break_action = self.get_observation(world_state)
        plt.close("all")
        # return self.obs.flatten()
        return [self.obs, 1 if self.allow_break_action else 0]

    def step(self, action):
        """
        Take an action in the environment and return the results.

        Args
            action: <np.array> [[action][material]]
                [[pickaxe,shovel,axe][diamond,golden]]

        Returns
            observation: <np.array> flattened array of obseravtion
            reward: <int> reward from taking action
            done: <bool> indicates terminal state
            info: <dict> dictionary of extra information
        """

        reward_delta = 0 # calling getWorldState() below loses our reward delta
        i_action = action[0]+action[1]*self.total_tool_type
        #find hotbar index of the action
        if action[1] == 1:
            #penalize gold tools and action log counts as incorrect for more than usage
            if self.golden_durability_dict[i_action]<1:
                reward_delta += 200
            else:
                if (i_action%self.total_tool_type==self.obs):
                    self.episode_action_log[i_action][0]+=1

            self.golden_durability_dict[i_action]-=1
        else:
            #penalize if using normal tools with special blocks and action log counts as incorrect
            if (self.obs >2 and self.obs<6):
                reward_delta +=200
            else:
                if (i_action%self.total_tool_type==self.obs):
                    self.episode_action_log[i_action][0]+=1

        #action logging
        self.episode_action_log[i_action][1] +=1






        # Get Action
        if self.allow_break_action:
        # switch tools, start digging

            self.agent_host.sendCommand('move 0');
            self.agent_host.sendCommand('hotbar.' + str(i_action + 1) + ' 1')
            self.agent_host.sendCommand('attack 1')
            # time.sleep(0.8)  # Allow steve to break the block

            # TODO
            # Add a while loop that lets us finish destroying the block
            # Or maybe this is a behavior it can learn on its own?
            # Otherwise, it might switch randomly, resetting the breaking

            while self.allow_break_action:
                time.sleep(0.2)
                world_state = self.agent_host.getWorldState()
                for r in world_state.rewards:   # these deltas will be lost otherwise
                    reward_delta += r.getValue()
                _, self.allow_break_action = self.get_observation(world_state)


        # just move forward if no block in front
        self.agent_host.sendCommand('attack 0')
        self.agent_host.sendCommand('move 1');
        time.sleep(0.5)

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
        self.initial_reward += reward_delta     # increment reward
        print("REWARD DELTA(# ticks taken): ", reward_delta)

        # NOTE
        # "Error: AgentHost::sendCommand : commands connection is not open. Is the mission running?"
        # The error above causes <RewardForTimeTaken> to register inconsistently
        # To be consistent, we keep track of seblocks/second*scalarlf.initial_reward internally

        if done:
            #reward is converted to blocks/minute
            reward += (int)(self.tunnel_len/self.initial_reward*20*60)   # add to reward so it can get passed out

            for action in range(self.total_tool_material*self.total_tool_type):
                self.action_log[action].append([self.episode_action_log[action][0],self.episode_action_log[action][1]])
                self.episode_returns=self.initial_reward
            print("END REWARD (blocks/minute)", reward)

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
        blocks = ['stone', 'stone', 'prismarine', 'dirt', 'dirt', 'clay',
                'planks', 'planks', 'log']
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
                                <InventoryItem slot="3" type="golden_pickaxe"/>
                                <InventoryItem slot="4" type="golden_shovel"/>
                                <InventoryItem slot="5" type="golden_axe"/>
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
                            <RewardForTimeTaken initialReward="0"  delta="1" density= "PER_TICK"/>
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
        """
        box = np.ones(self.log_frequency) / self.log_frequency
        returns_smooth = np.convolve(self.returns, box, mode='same')
        """
        plt.figure()
        plt.style.use('seaborn-darkgrid')
        palette = plt.get_cmap('Set1')
        plt.clf()
        plt.plot(list(range(len(self.returns))), self.returns,marker='',
                color=palette(1), linewidth=1, alpha=0.9)
        plt.title('BoringAI Continuous Sparse', loc='left', fontsize=12, fontweight=0,
                color='orange')
        plt.ylabel('Return')
        plt.xlabel('Episodes')
        plt.savefig('returns.png')
        with open('returns.txt', 'w') as f:
            for episode, value in enumerate( self.returns):
                f.write("{}\t{}\n".format(episode, value))
    def log_actions(self):
        """
        Log the number of actions taken correctly. e.g. axe to plank

        Args:
            action_log (dict): dict of array which for each block stores a
            number in the corresponding idx
        """
        plt.figure()
        plt.style.use('seaborn-darkgrid')
        palette = plt.get_cmap('Paired')
        num=-2
        for i_tool in range(self.total_tool_type*self.total_tool_material):
            if (num > 3):
                num=-1
            num+=2
            y_values = []

            stats=[0,0]
            for tmp in self.action_log[i_tool]:
                stats[0]+= tmp[0]
                stats[1]+= tmp[1]
                if stats[1]!=0:
                    y_values.append((stats[0]/stats[1]))
                else:
                    y_values.append(1)
            plt.plot(list(range(len(self.action_log[i_tool]))),y_values,
                    marker='', color=palette(num), linewidth=1.2, alpha=0.9,
                    label=self.tool_dict[i_tool])
        plt.legend(loc=2, ncol=2)
        plt.title("Tool Usage Statistics", loc='left', fontsize=12,
                fontweight=0, color='orange')
        plt.xlabel("Episode")
        plt.ylabel("Percentage Correctly Used")
        plt.savefig("toolstats.png")
        with open('toolstats.txt', 'w') as f:
            for episode in range(len(self.action_log[0])):
                f.write("{}\t".format(episode))
                for  i_tool in range(3):
                    f.write("{}\t{}\t".format(
                            self.action_log[i_tool][episode][0],self.action_log[i_tool][episode][1]))
                f.write("\n")

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
