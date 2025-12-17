# DQN

uv run levd.py --algorithm dqn --scenario defend_the_center --train_levels 0 1 --train_maps 1 --test_levels 2 3 4 --test_maps 1 --seed 50 --epoch 50 --step-per-collect 10 --device cuda --batch-size 64

Run: 50_20251215_001530

Epoch #1: 100001it [04:11, 397.10it/s, env_step=100000, len=77, loss=0.113, n/ep=0, n/st=16, rew=-0.20]                                                                   
Epoch #1: test_reward: 3.986100 ± 2.088371, best_reward: 3.986100 ± 2.088371 in #1
Epoch #2: 100001it [04:11, 398.17it/s, env_step=200000, len=107, loss=0.143, n/ep=0, n/st=16, rew=2.71]                                                                   
Epoch #2: test_reward: 4.100000 ± 2.046825, best_reward: 4.100000 ± 2.046825 in #2
Epoch #3: 100001it [04:11, 398.15it/s, env_step=300000, len=133, loss=0.132, n/ep=0, n/st=16, rew=6.65]                                                                   
Epoch #3: test_reward: 3.949300 ± 2.394101, best_reward: 4.100000 ± 2.046825 in #2
Epoch #4: 100001it [04:10, 398.53it/s, env_step=400000, len=66, loss=0.140, n/ep=0, n/st=16, rew=0.83]                                                                    
Epoch #4: test_reward: 4.009700 ± 2.150504, best_reward: 4.100000 ± 2.046825 in #2
Epoch #5: 100001it [04:10, 398.70it/s, env_step=500000, len=114, loss=0.147, n/ep=0, n/st=16, rew=4.56]                                                                   
Epoch #5: test_reward: 3.620100 ± 2.010665, best_reward: 4.100000 ± 2.046825 in #2
Epoch #6: 100001it [04:10, 398.52it/s, env_step=600000, len=82, loss=0.166, n/ep=0, n/st=16, rew=0.81]                                                                    
Epoch #6: test_reward: 3.176000 ± 1.823595, best_reward: 4.100000 ± 2.046825 in #2
Epoch #7: 100001it [04:10, 398.66it/s, env_step=700000, len=134, loss=0.149, n/ep=0, n/st=16, rew=4.69]                                                                   
Epoch #7: test_reward: 4.455600 ± 1.841334, best_reward: 4.455600 ± 1.841334 in #7
Epoch #8: 100001it [04:11, 398.35it/s, env_step=800000, len=72, loss=0.169, n/ep=0, n/st=16, rew=1.81]                                                                    
Epoch #8: test_reward: 4.267900 ± 1.784200, best_reward: 4.455600 ± 1.841334 in #7
Epoch #9: 100001it [04:10, 398.50it/s, env_step=900000, len=87, loss=0.148, n/ep=0, n/st=16, rew=2.84]                                                                    
Epoch #9: test_reward: 4.190200 ± 2.580575, best_reward: 4.455600 ± 1.841334 in #7
Epoch #10: 100001it [04:11, 398.03it/s, env_step=1000000, len=105, loss=0.157, n/ep=0, n/st=16, rew=2.76]                                                                 
Epoch #10: test_reward: 3.514100 ± 1.593501, best_reward: 4.455600 ± 1.841334 in #7
Epoch #11: 100001it [04:11, 398.15it/s, env_step=1100000, len=65, loss=0.155, n/ep=0, n/st=16, rew=2.86]                                                                  
Epoch #11: test_reward: 4.347600 ± 1.952994, best_reward: 4.455600 ± 1.841334 in #7
Epoch #12: 100001it [04:10, 398.77it/s, env_step=1200000, len=117, loss=0.153, n/ep=1, n/st=16, rew=3.68]                                                                 
Epoch #12: test_reward: 4.156700 ± 1.845816, best_reward: 4.455600 ± 1.841334 in #7
Epoch #13: 100001it [04:11, 398.32it/s, env_step=1300000, len=82, loss=0.158, n/ep=0, n/st=16, rew=1.85]                                                                  
Epoch #13: test_reward: 4.151700 ± 2.304922, best_reward: 4.455600 ± 1.841334 in #7
Epoch #14: 100001it [04:10, 398.50it/s, env_step=1400000, len=130, loss=0.147, n/ep=0, n/st=16, rew=4.70]                                                                 
Epoch #14: test_reward: 5.290100 ± 1.419676, best_reward: 5.290100 ± 1.419676 in #14
Epoch #15: 100001it [04:10, 398.57it/s, env_step=1500000, len=122, loss=0.149, n/ep=0, n/st=16, rew=5.67]                                                                 
Epoch #15: test_reward: 4.488300 ± 2.239057, best_reward: 5.290100 ± 1.419676 in #14
Epoch #16: 100001it [04:10, 398.76it/s, env_step=1600000, len=122, loss=0.144, n/ep=0, n/st=16, rew=4.68]                                                                 
Epoch #16: test_reward: 4.225300 ± 2.069253, best_reward: 5.290100 ± 1.419676 in #14
Epoch #17: 100001it [04:10, 398.85it/s, env_step=1700000, len=70, loss=0.138, n/ep=0, n/st=16, rew=3.84]                                                                  
Epoch #17: test_reward: 4.463000 ± 1.819707, best_reward: 5.290100 ± 1.419676 in #14
Epoch #18: 100001it [04:10, 399.13it/s, env_step=1800000, len=97, loss=0.134, n/ep=0, n/st=16, rew=3.77]                                                                  
Epoch #18: test_reward: 4.594400 ± 2.273420, best_reward: 5.290100 ± 1.419676 in #14
Epoch #19: 100001it [04:10, 398.85it/s, env_step=1900000, len=100, loss=0.148, n/ep=0, n/st=16, rew=2.70]                                                                 
Epoch #19: test_reward: 4.799000 ± 1.837666, best_reward: 5.290100 ± 1.419676 in #14
Epoch #20: 100001it [04:10, 399.12it/s, env_step=2000000, len=153, loss=0.141, n/ep=0, n/st=16, rew=7.20]                                                                 
Epoch #20: test_reward: 5.150700 ± 2.130433, best_reward: 5.290100 ± 1.419676 in #14
Epoch #21: 100001it [04:10, 399.43it/s, env_step=2100000, len=175, loss=0.125, n/ep=1, n/st=16, rew=5.68]                                                                 
Epoch #21: test_reward: 5.154400 ± 2.368918, best_reward: 5.290100 ± 1.419676 in #14
Epoch #22: 100001it [04:10, 398.75it/s, env_step=2200000, len=123, loss=0.124, n/ep=0, n/st=16, rew=3.63]                                                                 
Epoch #22: test_reward: 5.171900 ± 2.423126, best_reward: 5.290100 ± 1.419676 in #14
Epoch #23: 100001it [04:10, 398.85it/s, env_step=2300000, len=126, loss=0.121, n/ep=0, n/st=16, rew=4.71]                                                                 
Epoch #23: test_reward: 5.142900 ± 2.163707, best_reward: 5.290100 ± 1.419676 in #14
Epoch #24: 100001it [04:10, 398.99it/s, env_step=2400000, len=138, loss=0.124, n/ep=0, n/st=16, rew=6.70]                                                                 
Epoch #24: test_reward: 4.968200 ± 2.026281, best_reward: 5.290100 ± 1.419676 in #14
Epoch #25: 100001it [04:10, 398.67it/s, env_step=2500000, len=96, loss=0.125, n/ep=0, n/st=16, rew=3.76]                                                                  
Epoch #25: test_reward: 4.638300 ± 2.484722, best_reward: 5.290100 ± 1.419676 in #14
Epoch #26: 100001it [04:10, 398.57it/s, env_step=2600000, len=116, loss=0.104, n/ep=0, n/st=16, rew=4.69]                                                                 
Epoch #26: test_reward: 4.958500 ± 2.339967, best_reward: 5.290100 ± 1.419676 in #14
Epoch #27: 100001it [04:10, 399.13it/s, env_step=2700000, len=152, loss=0.111, n/ep=0, n/st=16, rew=6.47]                                                                 
Epoch #27: test_reward: 4.662500 ± 2.189995, best_reward: 5.290100 ± 1.419676 in #14
Epoch #28: 100001it [04:10, 399.07it/s, env_step=2800000, len=130, loss=0.105, n/ep=0, n/st=16, rew=5.71]                                                                 
Epoch #28: test_reward: 5.262300 ± 2.618870, best_reward: 5.290100 ± 1.419676 in #14
Epoch #29: 100001it [04:10, 399.12it/s, env_step=2900000, len=124, loss=0.109, n/ep=0, n/st=16, rew=5.67]                                                                 
Epoch #29: test_reward: 4.947800 ± 2.380813, best_reward: 5.290100 ± 1.419676 in #14
Epoch #30: 100001it [04:10, 398.71it/s, env_step=3000000, len=95, loss=0.109, n/ep=0, n/st=16, rew=3.72]                                                                  
Epoch #30: test_reward: 4.392700 ± 1.878078, best_reward: 5.290100 ± 1.419676 in #14
Epoch #31: 100001it [04:10, 399.24it/s, env_step=3100000, len=105, loss=0.112, n/ep=0, n/st=16, rew=4.54]                                                                 
Epoch #31: test_reward: 5.252700 ± 2.452435, best_reward: 5.290100 ± 1.419676 in #14
Epoch #32: 100001it [04:10, 398.69it/s, env_step=3200000, len=102, loss=0.109, n/ep=0, n/st=16, rew=2.74]                                                                 
Epoch #32: test_reward: 5.003800 ± 2.140018, best_reward: 5.290100 ± 1.419676 in #14
Epoch #33: 100001it [04:10, 398.41it/s, env_step=3300000, len=202, loss=0.109, n/ep=0, n/st=16, rew=7.65]                                                                 
Epoch #33: test_reward: 4.705100 ± 2.115494, best_reward: 5.290100 ± 1.419676 in #14
Epoch #34: 100001it [04:11, 398.33it/s, env_step=3400000, len=119, loss=0.102, n/ep=0, n/st=16, rew=3.68]                                                                 
Epoch #34: test_reward: 4.869700 ± 2.576916, best_reward: 5.290100 ± 1.419676 in #14
Epoch #35: 100001it [04:11, 397.98it/s, env_step=3500000, len=107, loss=0.102, n/ep=0, n/st=16, rew=4.69]                                                                 
Epoch #35: test_reward: 5.247800 ± 2.414711, best_reward: 5.290100 ± 1.419676 in #14
Epoch #36: 100001it [04:11, 397.87it/s, env_step=3600000, len=140, loss=0.108, n/ep=0, n/st=16, rew=6.43]                                                                 
Epoch #36: test_reward: 5.470600 ± 2.003512, best_reward: 5.470600 ± 2.003512 in #36
Epoch #37: 100001it [04:11, 398.06it/s, env_step=3700000, len=103, loss=0.100, n/ep=0, n/st=16, rew=2.67]                                                                 
Epoch #37: test_reward: 4.690700 ± 2.301534, best_reward: 5.470600 ± 2.003512 in #36
Epoch #38: 100001it [04:11, 398.24it/s, env_step=3800000, len=79, loss=0.112, n/ep=0, n/st=16, rew=2.77]                                                                  
Epoch #38: test_reward: 5.247400 ± 2.121963, best_reward: 5.470600 ± 2.003512 in #36
Epoch #39: 100001it [04:10, 398.45it/s, env_step=3900000, len=107, loss=0.110, n/ep=0, n/st=16, rew=3.77]                                                                 
Epoch #39: test_reward: 5.131500 ± 2.304142, best_reward: 5.470600 ± 2.003512 in #36
Epoch #40: 100001it [04:11, 398.32it/s, env_step=4000000, len=163, loss=0.106, n/ep=0, n/st=16, rew=7.48]                                                                 
Epoch #40: test_reward: 4.810200 ± 2.193380, best_reward: 5.470600 ± 2.003512 in #36
Epoch #41: 100001it [04:11, 398.06it/s, env_step=4100000, len=98, loss=0.109, n/ep=0, n/st=16, rew=2.69]                                                                  
Epoch #41: test_reward: 4.964400 ± 2.514974, best_reward: 5.470600 ± 2.003512 in #36
Epoch #42: 100001it [04:11, 398.01it/s, env_step=4200000, len=72, loss=0.114, n/ep=0, n/st=16, rew=1.78]                                                                  
Epoch #42: test_reward: 4.085400 ± 2.197003, best_reward: 5.470600 ± 2.003512 in #36
Epoch #43: 100001it [04:11, 398.13it/s, env_step=4300000, len=139, loss=0.112, n/ep=0, n/st=16, rew=5.67]                                                                 
Epoch #43: test_reward: 5.409000 ± 2.105694, best_reward: 5.470600 ± 2.003512 in #36
Epoch #44: 100001it [04:11, 397.86it/s, env_step=4400000, len=125, loss=0.102, n/ep=0, n/st=16, rew=4.69]                                                                 
Epoch #44: test_reward: 5.201800 ± 2.146897, best_reward: 5.470600 ± 2.003512 in #36
Epoch #45: 100001it [04:11, 398.00it/s, env_step=4500000, len=129, loss=0.104, n/ep=0, n/st=16, rew=5.66]                                                                 
Epoch #45: test_reward: 5.525800 ± 1.981041, best_reward: 5.525800 ± 1.981041 in #45
Epoch #46: 100001it [04:11, 397.97it/s, env_step=4600000, len=115, loss=0.108, n/ep=0, n/st=16, rew=3.67]                                                                 
Epoch #46: test_reward: 5.268600 ± 2.176910, best_reward: 5.525800 ± 1.981041 in #45
Epoch #47: 100001it [04:11, 397.74it/s, env_step=4700000, len=148, loss=0.116, n/ep=0, n/st=16, rew=6.67]                                                                 
Epoch #47: test_reward: 4.779700 ± 2.347568, best_reward: 5.525800 ± 1.981041 in #45
Epoch #48: 100001it [04:11, 398.09it/s, env_step=4800000, len=111, loss=0.112, n/ep=0, n/st=16, rew=3.68]                                                                 
Epoch #48: test_reward: 4.583400 ± 1.917137, best_reward: 5.525800 ± 1.981041 in #45
Epoch #49: 100001it [04:11, 397.93it/s, env_step=4900000, len=152, loss=0.114, n/ep=0, n/st=16, rew=7.66]                                                                 
Epoch #49: test_reward: 5.285200 ± 1.631498, best_reward: 5.525800 ± 1.981041 in #45
Epoch #50: 100001it [04:11, 398.28it/s, env_step=5000000, len=146, loss=0.109, n/ep=0, n/st=16, rew=5.69]                                                                 
Epoch #50: test_reward: 5.226400 ± 1.823205, best_reward: 5.525800 ± 1.981041 in #45
{'best_result': '5.53 ± 1.98',
 'best_reward': 5.525800000000001,
 'duration': '12751.10s',
 'test_episode': 5100,
 'test_speed': '2872.18 step/s',
 'test_step': 579971,
 'test_time': '201.93s',
 'train_episode': 40971,
 'train_speed': '398.43 step/s',
 'train_step': 5000000,
 'train_time/collector': '1810.45s',
 'train_time/model': '10738.72s'}


# C51

uv run levd.py --algorithm c51 --scenario defend_the_center --train_levels 0 1 --train_maps 1 --test_lev
els 2 3 4 --test_maps 1 --seed 50 --epoch 50 --step-per-collect 10 --device cuda --batch-size 64

Run: 50_20251215_091719

Epoch #1: 100001it [03:52, 429.72it/s, env_step=100000, len=98, loss=3.056, n/ep=0, n/st=16, rew=1.70]                                                                    
Epoch #1: test_reward: 2.413400 ± 1.536213, best_reward: 2.413400 ± 1.536213 in #1
Epoch #2: 100001it [03:52, 430.30it/s, env_step=200000, len=93, loss=2.843, n/ep=0, n/st=16, rew=2.57]                                                                    
Epoch #2: test_reward: 3.632300 ± 2.215958, best_reward: 3.632300 ± 2.215958 in #2
Epoch #3: 100001it [03:52, 430.71it/s, env_step=300000, len=81, loss=2.708, n/ep=0, n/st=16, rew=-0.24]                                                                   
Epoch #3: test_reward: 3.805600 ± 1.763747, best_reward: 3.805600 ± 1.763747 in #3
Epoch #4: 100001it [03:52, 429.90it/s, env_step=400000, len=99, loss=2.726, n/ep=0, n/st=16, rew=1.71]                                                                    
Epoch #4: test_reward: 3.799500 ± 1.804376, best_reward: 3.805600 ± 1.763747 in #3
Epoch #5: 100001it [03:52, 429.45it/s, env_step=500000, len=123, loss=2.708, n/ep=0, n/st=16, rew=3.68]                                                                   
Epoch #5: test_reward: 3.942500 ± 1.790930, best_reward: 3.942500 ± 1.790930 in #5
Epoch #6: 100001it [03:52, 430.16it/s, env_step=600000, len=88, loss=2.669, n/ep=1, n/st=16, rew=1.74]                                                                    
Epoch #6: test_reward: 4.354500 ± 1.751360, best_reward: 4.354500 ± 1.751360 in #6
Epoch #7: 100001it [03:53, 428.67it/s, env_step=700000, len=116, loss=2.692, n/ep=0, n/st=16, rew=4.65]                                                                   
Epoch #7: test_reward: 4.399400 ± 1.649787, best_reward: 4.399400 ± 1.649787 in #7
Epoch #8: 100001it [03:53, 428.65it/s, env_step=800000, len=114, loss=2.696, n/ep=0, n/st=16, rew=5.72]                                                                   
Epoch #8: test_reward: 4.171000 ± 1.576824, best_reward: 4.399400 ± 1.649787 in #7
Epoch #9: 100001it [03:53, 427.91it/s, env_step=900000, len=118, loss=2.730, n/ep=0, n/st=16, rew=5.74]                                                                   
Epoch #9: test_reward: 3.908900 ± 1.924346, best_reward: 4.399400 ± 1.649787 in #7
Epoch #10: 100001it [03:53, 428.44it/s, env_step=1000000, len=115, loss=2.716, n/ep=0, n/st=16, rew=4.71]                                                                 
Epoch #10: test_reward: 4.026000 ± 1.687076, best_reward: 4.399400 ± 1.649787 in #7
Epoch #11: 100001it [03:53, 429.11it/s, env_step=1100000, len=113, loss=2.733, n/ep=0, n/st=16, rew=3.71]                                                                 
Epoch #11: test_reward: 4.842700 ± 1.929758, best_reward: 4.842700 ± 1.929758 in #11
Epoch #12: 100001it [03:53, 428.74it/s, env_step=1200000, len=135, loss=2.719, n/ep=0, n/st=16, rew=5.66]                                                                 
Epoch #12: test_reward: 4.136700 ± 1.907249, best_reward: 4.842700 ± 1.929758 in #11
Epoch #13: 100001it [03:52, 429.79it/s, env_step=1300000, len=115, loss=2.703, n/ep=1, n/st=16, rew=3.66]                                                                 
Epoch #13: test_reward: 4.676100 ± 2.036368, best_reward: 4.842700 ± 1.929758 in #11
Epoch #14: 100001it [03:53, 429.06it/s, env_step=1400000, len=87, loss=2.693, n/ep=0, n/st=16, rew=3.79]                                                                  
Epoch #14: test_reward: 4.278100 ± 2.054755, best_reward: 4.842700 ± 1.929758 in #11
Epoch #15: 100001it [03:53, 428.54it/s, env_step=1500000, len=206, loss=2.671, n/ep=0, n/st=16, rew=7.67]                                                                 
Epoch #15: test_reward: 3.644000 ± 1.879499, best_reward: 4.842700 ± 1.929758 in #11
Epoch #16: 100001it [03:53, 428.72it/s, env_step=1600000, len=101, loss=2.639, n/ep=1, n/st=16, rew=3.71]                                                                 
Epoch #16: test_reward: 3.816400 ± 1.744097, best_reward: 4.842700 ± 1.929758 in #11
Epoch #17: 100001it [03:52, 430.06it/s, env_step=1700000, len=148, loss=2.623, n/ep=0, n/st=16, rew=5.67]                                                                 
Epoch #17: test_reward: 3.441100 ± 1.949236, best_reward: 4.842700 ± 1.929758 in #11
Epoch #18: 100001it [03:52, 429.44it/s, env_step=1800000, len=115, loss=2.633, n/ep=0, n/st=16, rew=3.80]                                                                 
Epoch #18: test_reward: 3.756100 ± 2.037352, best_reward: 4.842700 ± 1.929758 in #11
Epoch #19: 100001it [03:52, 430.37it/s, env_step=1900000, len=133, loss=2.597, n/ep=1, n/st=16, rew=3.71]                                                                 
Epoch #19: test_reward: 4.422900 ± 2.349307, best_reward: 4.842700 ± 1.929758 in #11
Epoch #20: 100001it [03:53, 428.85it/s, env_step=2000000, len=127, loss=2.671, n/ep=0, n/st=16, rew=5.10]                                                                 
Epoch #20: test_reward: 4.106100 ± 2.005208, best_reward: 4.842700 ± 1.929758 in #11
Epoch #21: 100001it [03:53, 429.09it/s, env_step=2100000, len=159, loss=2.610, n/ep=0, n/st=16, rew=6.35]                                                                 
Epoch #21: test_reward: 5.025800 ± 2.321471, best_reward: 5.025800 ± 2.321471 in #21
Epoch #22: 100001it [03:53, 428.87it/s, env_step=2200000, len=103, loss=2.610, n/ep=0, n/st=16, rew=3.71]                                                                 
Epoch #22: test_reward: 4.300400 ± 1.940674, best_reward: 5.025800 ± 2.321471 in #21
Epoch #23: 100001it [03:52, 430.54it/s, env_step=2300000, len=150, loss=2.602, n/ep=0, n/st=16, rew=7.67]                                                                 
Epoch #23: test_reward: 5.397500 ± 1.825675, best_reward: 5.397500 ± 1.825675 in #23
Epoch #24: 100001it [03:53, 428.30it/s, env_step=2400000, len=154, loss=2.598, n/ep=0, n/st=16, rew=6.66]                                                                 
Epoch #24: test_reward: 4.156700 ± 1.941183, best_reward: 5.397500 ± 1.825675 in #23
Epoch #25: 100001it [03:52, 430.18it/s, env_step=2500000, len=187, loss=2.597, n/ep=0, n/st=16, rew=8.68]                                                                 
Epoch #25: test_reward: 5.044200 ± 1.944447, best_reward: 5.397500 ± 1.825675 in #23
Epoch #26: 100001it [03:53, 429.01it/s, env_step=2600000, len=76, loss=2.595, n/ep=1, n/st=16, rew=0.82]                                                                  
Epoch #26: test_reward: 3.889400 ± 1.830298, best_reward: 5.397500 ± 1.825675 in #23
Epoch #27: 100001it [03:53, 428.54it/s, env_step=2700000, len=125, loss=2.620, n/ep=0, n/st=16, rew=5.58]                                                                 
Epoch #27: test_reward: 4.944900 ± 2.132863, best_reward: 5.397500 ± 1.825675 in #23
Epoch #28: 100001it [03:52, 430.05it/s, env_step=2800000, len=144, loss=2.573, n/ep=0, n/st=16, rew=4.71]                                                                 
Epoch #28: test_reward: 4.637900 ± 1.810450, best_reward: 5.397500 ± 1.825675 in #23
Epoch #29: 100001it [03:53, 428.58it/s, env_step=2900000, len=120, loss=2.586, n/ep=0, n/st=16, rew=2.65]                                                                 
Epoch #29: test_reward: 4.233500 ± 2.202457, best_reward: 5.397500 ± 1.825675 in #23
Epoch #30: 100001it [03:55, 424.62it/s, env_step=3000000, len=109, loss=2.589, n/ep=1, n/st=16, rew=4.68]                                                                 
Epoch #30: test_reward: 4.780900 ± 1.689016, best_reward: 5.397500 ± 1.825675 in #23
Epoch #31: 100001it [03:55, 425.23it/s, env_step=3100000, len=111, loss=2.627, n/ep=0, n/st=16, rew=1.69]                                                                 
Epoch #31: test_reward: 4.312500 ± 2.118517, best_reward: 5.397500 ± 1.825675 in #23
Epoch #32: 100001it [03:56, 422.20it/s, env_step=3200000, len=182, loss=2.602, n/ep=0, n/st=16, rew=7.69]                                                                 
Epoch #32: test_reward: 3.700100 ± 1.958369, best_reward: 5.397500 ± 1.825675 in #23
Epoch #33: 100001it [03:59, 417.85it/s, env_step=3300000, len=107, loss=2.614, n/ep=0, n/st=16, rew=2.67]                                                                 
Epoch #33: test_reward: 4.650500 ± 1.941435, best_reward: 5.397500 ± 1.825675 in #23
Epoch #34: 100001it [03:59, 418.28it/s, env_step=3400000, len=129, loss=2.573, n/ep=0, n/st=16, rew=4.65]                                                                 
Epoch #34: test_reward: 3.542600 ± 1.848106, best_reward: 5.397500 ± 1.825675 in #23
Epoch #35: 100001it [03:54, 426.12it/s, env_step=3500000, len=141, loss=2.617, n/ep=0, n/st=16, rew=6.68]                                                                 
Epoch #35: test_reward: 3.738600 ± 1.907442, best_reward: 5.397500 ± 1.825675 in #23
Epoch #36: 100001it [03:54, 426.13it/s, env_step=3600000, len=99, loss=2.605, n/ep=0, n/st=16, rew=3.74]                                                                  
Epoch #36: test_reward: 4.322400 ± 2.078240, best_reward: 5.397500 ± 1.825675 in #23
Epoch #37: 100001it [03:55, 425.51it/s, env_step=3700000, len=89, loss=2.596, n/ep=0, n/st=16, rew=2.56]                                                                  
Epoch #37: test_reward: 3.375800 ± 1.845233, best_reward: 5.397500 ± 1.825675 in #23
Epoch #38: 100001it [03:56, 423.38it/s, env_step=3800000, len=94, loss=2.624, n/ep=0, n/st=16, rew=2.73]                                                                  
Epoch #38: test_reward: 3.582100 ± 2.043941, best_reward: 5.397500 ± 1.825675 in #23
Epoch #39: 100001it [03:57, 420.53it/s, env_step=3900000, len=164, loss=2.601, n/ep=0, n/st=16, rew=5.63]                                                                 
Epoch #39: test_reward: 4.072500 ± 1.768068, best_reward: 5.397500 ± 1.825675 in #23
Epoch #40: 100001it [03:55, 425.41it/s, env_step=4000000, len=130, loss=2.620, n/ep=0, n/st=16, rew=5.12]                                                                 
Epoch #40: test_reward: 3.461800 ± 2.047821, best_reward: 5.397500 ± 1.825675 in #23
Epoch #41: 100001it [03:56, 423.67it/s, env_step=4100000, len=152, loss=2.617, n/ep=0, n/st=16, rew=6.66]                                                                 
Epoch #41: test_reward: 3.948300 ± 1.774596, best_reward: 5.397500 ± 1.825675 in #23
Epoch #42: 100001it [03:55, 423.75it/s, env_step=4200000, len=146, loss=2.634, n/ep=0, n/st=16, rew=6.70]                                                                 
Epoch #42: test_reward: 3.974800 ± 1.860196, best_reward: 5.397500 ± 1.825675 in #23
Epoch #43: 100001it [03:55, 423.94it/s, env_step=4300000, len=124, loss=2.637, n/ep=0, n/st=16, rew=3.66]                                                                 
Epoch #43: test_reward: 3.809400 ± 1.659309, best_reward: 5.397500 ± 1.825675 in #23
Epoch #44: 100001it [03:56, 422.50it/s, env_step=4400000, len=123, loss=2.624, n/ep=0, n/st=16, rew=4.18]                                                                 
Epoch #44: test_reward: 4.190300 ± 1.630346, best_reward: 5.397500 ± 1.825675 in #23
Epoch #45: 100001it [03:55, 424.32it/s, env_step=4500000, len=124, loss=2.607, n/ep=0, n/st=16, rew=2.67]                                                                 
Epoch #45: test_reward: 3.718900 ± 1.508858, best_reward: 5.397500 ± 1.825675 in #23
Epoch #46: 100001it [03:55, 425.05it/s, env_step=4600000, len=107, loss=2.623, n/ep=0, n/st=16, rew=1.80]                                                                 
Epoch #46: test_reward: 3.874600 ± 1.971693, best_reward: 5.397500 ± 1.825675 in #23
Epoch #47: 100001it [03:55, 425.14it/s, env_step=4700000, len=115, loss=2.653, n/ep=0, n/st=16, rew=1.69]                                                                 
Epoch #47: test_reward: 3.916800 ± 1.813660, best_reward: 5.397500 ± 1.825675 in #23
Epoch #48: 100001it [03:55, 424.09it/s, env_step=4800000, len=110, loss=2.628, n/ep=0, n/st=16, rew=4.52]                                                                 
Epoch #48: test_reward: 3.694500 ± 1.737410, best_reward: 5.397500 ± 1.825675 in #23
Epoch #49: 100001it [03:54, 425.78it/s, env_step=4900000, len=121, loss=2.615, n/ep=0, n/st=16, rew=4.65]                                                                 
Epoch #49: test_reward: 3.846900 ± 1.809968, best_reward: 5.397500 ± 1.825675 in #23
Epoch #50: 100001it [03:55, 425.07it/s, env_step=5000000, len=122, loss=2.627, n/ep=0, n/st=16, rew=3.51]                                                                 
Epoch #50: test_reward: 3.809900 ± 1.955453, best_reward: 5.397500 ± 1.825675 in #23
{'best_result': '5.40 ± 1.83',
 'best_reward': 5.397500000000003,
 'duration': '11915.64s',
 'test_episode': 5100,
 'test_speed': '2794.66 step/s',
 'test_step': 569475,
 'test_time': '203.77s',
 'train_episode': 40931,
 'train_speed': '426.92 step/s',
 'train_step': 5000000,
 'train_time/collector': '1885.08s',
 'train_time/model': '9826.79s'}

# Rainbow

uv run levd.py --algorithm rainbow --scenario defend_the_center --train_levels 0 1 --train_maps 1 --test
_levels 2 3 4 --test_maps 1 --seed 50 --epoch 50 --step-per-collect 10 --device cuda --batch-size 64

Epoch #1: 100001it [04:12, 395.91it/s, env_step=100000, len=71, loss=2.372, n/ep=0, n/st=16, rew=-0.20]                                                                   
Epoch #1: test_reward: 4.631200 ± 1.830832, best_reward: 4.631200 ± 1.830832 in #1
Epoch #2: 100001it [04:13, 394.78it/s, env_step=200000, len=105, loss=2.114, n/ep=0, n/st=16, rew=2.69]                                                                   
Epoch #2: test_reward: 4.740900 ± 1.820399, best_reward: 4.740900 ± 1.820399 in #2
Epoch #3: 100001it [04:13, 394.92it/s, env_step=300000, len=66, loss=2.028, n/ep=1, n/st=16, rew=-0.21]                                                                   
Epoch #3: test_reward: 4.149300 ± 1.377105, best_reward: 4.740900 ± 1.820399 in #2
Epoch #4: 100001it [04:13, 394.42it/s, env_step=400000, len=125, loss=1.968, n/ep=0, n/st=16, rew=3.67]                                                                   
Epoch #4: test_reward: 4.814700 ± 1.609916, best_reward: 4.814700 ± 1.609916 in #4
Epoch #5: 100001it [04:14, 393.56it/s, env_step=500000, len=129, loss=1.996, n/ep=0, n/st=16, rew=4.68]                                                                   
Epoch #5: test_reward: 4.839200 ± 1.699647, best_reward: 4.839200 ± 1.699647 in #5
Epoch #6: 100001it [04:13, 394.29it/s, env_step=600000, len=78, loss=1.995, n/ep=0, n/st=16, rew=1.81]                                                                    
Epoch #6: test_reward: 4.618600 ± 1.719845, best_reward: 4.839200 ± 1.699647 in #5
Epoch #7: 100001it [04:13, 394.07it/s, env_step=700000, len=127, loss=2.033, n/ep=0, n/st=16, rew=4.63]                                                                   
Epoch #7: test_reward: 4.367800 ± 1.570478, best_reward: 4.839200 ± 1.699647 in #5
Epoch #8: 100001it [04:14, 393.68it/s, env_step=800000, len=105, loss=2.014, n/ep=0, n/st=16, rew=2.77]                                                                   
Epoch #8: test_reward: 5.064700 ± 1.896698, best_reward: 5.064700 ± 1.896698 in #8
Epoch #9: 100001it [04:13, 394.07it/s, env_step=900000, len=142, loss=2.009, n/ep=0, n/st=16, rew=6.69]                                                                   
Epoch #9: test_reward: 4.422500 ± 2.064484, best_reward: 5.064700 ± 1.896698 in #8
Epoch #10: 100001it [04:13, 393.98it/s, env_step=1000000, len=99, loss=1.975, n/ep=1, n/st=16, rew=3.71]                                                                  
Epoch #10: test_reward: 4.873900 ± 1.741529, best_reward: 5.064700 ± 1.896698 in #8
Epoch #11: 100001it [04:14, 393.49it/s, env_step=1100000, len=152, loss=1.977, n/ep=0, n/st=16, rew=4.72]                                                                 
Epoch #11: test_reward: 4.867200 ± 1.517752, best_reward: 5.064700 ± 1.896698 in #8
Epoch #12: 100001it [04:13, 393.99it/s, env_step=1200000, len=177, loss=2.004, n/ep=0, n/st=16, rew=6.68]                                                                 
Epoch #12: test_reward: 5.092700 ± 1.657377, best_reward: 5.092700 ± 1.657377 in #12
Epoch #13: 100001it [04:13, 394.01it/s, env_step=1300000, len=117, loss=1.959, n/ep=0, n/st=16, rew=5.66]                                                                 
Epoch #13: test_reward: 5.159100 ± 1.781310, best_reward: 5.159100 ± 1.781310 in #13
Epoch #14: 100001it [04:13, 394.13it/s, env_step=1400000, len=170, loss=1.959, n/ep=0, n/st=16, rew=8.52]                                                                 
Epoch #14: test_reward: 4.630400 ± 1.612432, best_reward: 5.159100 ± 1.781310 in #13
Epoch #15: 100001it [04:14, 393.54it/s, env_step=1500000, len=177, loss=1.962, n/ep=0, n/st=16, rew=7.69]                                                                 
Epoch #15: test_reward: 5.011500 ± 1.693376, best_reward: 5.159100 ± 1.781310 in #13
Epoch #16: 100001it [04:14, 393.50it/s, env_step=1600000, len=141, loss=1.830, n/ep=1, n/st=16, rew=5.69]                                                                 
Epoch #16: test_reward: 4.765900 ± 1.913094, best_reward: 5.159100 ± 1.781310 in #13
Epoch #17: 100001it [04:13, 394.06it/s, env_step=1700000, len=168, loss=1.944, n/ep=0, n/st=16, rew=7.69]                                                                 
Epoch #17: test_reward: 5.149700 ± 2.188755, best_reward: 5.159100 ± 1.781310 in #13
Epoch #18: 100001it [04:13, 394.28it/s, env_step=1800000, len=116, loss=1.795, n/ep=0, n/st=16, rew=4.72]                                                                 
Epoch #18: test_reward: 4.696000 ± 1.819265, best_reward: 5.159100 ± 1.781310 in #13
Epoch #19: 100001it [04:13, 394.23it/s, env_step=1900000, len=162, loss=1.831, n/ep=0, n/st=16, rew=5.68]                                                                 
Epoch #19: test_reward: 5.512800 ± 2.270832, best_reward: 5.512800 ± 2.270832 in #19
Epoch #20: 100001it [04:13, 394.65it/s, env_step=2000000, len=182, loss=1.732, n/ep=0, n/st=16, rew=7.49]                                                                 
Epoch #20: test_reward: 5.092300 ± 1.783134, best_reward: 5.512800 ± 2.270832 in #19
Epoch #21: 100001it [04:13, 394.41it/s, env_step=2100000, len=159, loss=1.770, n/ep=0, n/st=16, rew=6.44]                                                                 
Epoch #21: test_reward: 5.407500 ± 1.999195, best_reward: 5.512800 ± 2.270832 in #19
Epoch #22: 100001it [04:14, 393.33it/s, env_step=2200000, len=114, loss=1.762, n/ep=2, n/st=16, rew=5.73]                                                                 
Epoch #22: test_reward: 5.606500 ± 1.929937, best_reward: 5.606500 ± 1.929937 in #22
Epoch #23: 100001it [04:14, 393.12it/s, env_step=2300000, len=152, loss=1.785, n/ep=0, n/st=16, rew=6.66]                                                                 
Epoch #23: test_reward: 5.038700 ± 1.831838, best_reward: 5.606500 ± 1.929937 in #22
Epoch #24: 100001it [04:14, 393.42it/s, env_step=2400000, len=137, loss=1.798, n/ep=0, n/st=16, rew=6.69]                                                                 
Epoch #24: test_reward: 4.876300 ± 2.339362, best_reward: 5.606500 ± 1.929937 in #22
Epoch #25: 100001it [04:13, 394.04it/s, env_step=2500000, len=71, loss=1.810, n/ep=1, n/st=16, rew=1.81]                                                                  
Epoch #25: test_reward: 5.171800 ± 1.853235, best_reward: 5.606500 ± 1.929937 in #22
Epoch #26: 100001it [04:14, 393.61it/s, env_step=2600000, len=78, loss=1.799, n/ep=0, n/st=16, rew=2.77]                                                                  
Epoch #26: test_reward: 5.230500 ± 1.936852, best_reward: 5.606500 ± 1.929937 in #22
Epoch #27: 100001it [04:14, 393.55it/s, env_step=2700000, len=183, loss=1.816, n/ep=0, n/st=16, rew=6.63]                                                                 
Epoch #27: test_reward: 5.518700 ± 1.914176, best_reward: 5.606500 ± 1.929937 in #22
Epoch #28: 100001it [04:14, 392.95it/s, env_step=2800000, len=130, loss=1.779, n/ep=0, n/st=16, rew=4.54]                                                                 
Epoch #28: test_reward: 5.045000 ± 1.831325, best_reward: 5.606500 ± 1.929937 in #22
Epoch #29: 100001it [04:14, 392.73it/s, env_step=2900000, len=104, loss=1.769, n/ep=0, n/st=16, rew=2.75]                                                                 
Epoch #29: test_reward: 5.100800 ± 1.896445, best_reward: 5.606500 ± 1.929937 in #22
Epoch #30: 100001it [04:14, 393.60it/s, env_step=3000000, len=130, loss=1.817, n/ep=0, n/st=16, rew=4.63]                                                                 
Epoch #30: test_reward: 5.121900 ± 2.281257, best_reward: 5.606500 ± 1.929937 in #22
Epoch #31: 100001it [04:14, 393.45it/s, env_step=3100000, len=96, loss=1.713, n/ep=0, n/st=16, rew=1.77]                                                                  
Epoch #31: test_reward: 4.846500 ± 2.067840, best_reward: 5.606500 ± 1.929937 in #22
Epoch #32: 100001it [04:14, 392.42it/s, env_step=3200000, len=77, loss=1.762, n/ep=0, n/st=16, rew=1.87]                                                                  
Epoch #32: test_reward: 5.305000 ± 2.053555, best_reward: 5.606500 ± 1.929937 in #22
Epoch #33: 100001it [04:15, 391.05it/s, env_step=3300000, len=116, loss=1.772, n/ep=0, n/st=16, rew=4.70]                                                                 
Epoch #33: test_reward: 5.203200 ± 2.092410, best_reward: 5.606500 ± 1.929937 in #22
Epoch #34: 100001it [04:16, 390.18it/s, env_step=3400000, len=136, loss=1.867, n/ep=1, n/st=16, rew=7.71]                                                                 
Epoch #34: test_reward: 4.929800 ± 2.025145, best_reward: 5.606500 ± 1.929937 in #22
Epoch #35: 100001it [04:19, 385.29it/s, env_step=3500000, len=163, loss=1.785, n/ep=0, n/st=16, rew=6.66]                                                                 
Epoch #35: test_reward: 5.299400 ± 2.029282, best_reward: 5.606500 ± 1.929937 in #22
Epoch #36: 100001it [04:18, 386.88it/s, env_step=3600000, len=114, loss=1.821, n/ep=0, n/st=16, rew=2.76]                                                                 
Epoch #36: test_reward: 5.157400 ± 2.309948, best_reward: 5.606500 ± 1.929937 in #22
Epoch #37: 100001it [04:19, 385.55it/s, env_step=3700000, len=97, loss=1.733, n/ep=0, n/st=16, rew=4.78]                                                                  
Epoch #37: test_reward: 5.553700 ± 2.447627, best_reward: 5.606500 ± 1.929937 in #22
Epoch #38: 100001it [04:19, 385.73it/s, env_step=3800000, len=145, loss=1.770, n/ep=0, n/st=16, rew=5.68]                                                                 
Epoch #38: test_reward: 5.270500 ± 2.207955, best_reward: 5.606500 ± 1.929937 in #22
Epoch #39: 100001it [04:18, 386.29it/s, env_step=3900000, len=111, loss=1.794, n/ep=0, n/st=16, rew=2.75]                                                                 
Epoch #39: test_reward: 5.221900 ± 1.714075, best_reward: 5.606500 ± 1.929937 in #22
Epoch #40: 100001it [04:16, 390.36it/s, env_step=4000000, len=118, loss=1.732, n/ep=0, n/st=16, rew=3.66]                                                                 
Epoch #40: test_reward: 5.027700 ± 2.202725, best_reward: 5.606500 ± 1.929937 in #22
Epoch #41: 100001it [04:16, 389.90it/s, env_step=4100000, len=195, loss=1.767, n/ep=0, n/st=16, rew=7.66]                                                                 
Epoch #41: test_reward: 5.447300 ± 2.294085, best_reward: 5.606500 ± 1.929937 in #22
Epoch #42: 100001it [04:14, 393.16it/s, env_step=4200000, len=170, loss=1.763, n/ep=0, n/st=16, rew=5.70]                                                                 
Epoch #42: test_reward: 5.567800 ± 1.944712, best_reward: 5.606500 ± 1.929937 in #22
Epoch #43: 100001it [04:13, 394.33it/s, env_step=4300000, len=120, loss=1.770, n/ep=0, n/st=16, rew=3.69]                                                                 
Epoch #43: test_reward: 5.309800 ± 2.334682, best_reward: 5.606500 ± 1.929937 in #22
Epoch #44: 100001it [04:13, 394.45it/s, env_step=4400000, len=194, loss=1.764, n/ep=0, n/st=16, rew=7.66]                                                                 
Epoch #44: test_reward: 4.980000 ± 2.212699, best_reward: 5.606500 ± 1.929937 in #22
Epoch #45: 100001it [04:13, 394.98it/s, env_step=4500000, len=141, loss=1.719, n/ep=0, n/st=16, rew=6.23]                                                                 
Epoch #45: test_reward: 4.795400 ± 2.164240, best_reward: 5.606500 ± 1.929937 in #22
Epoch #46: 100001it [04:13, 394.88it/s, env_step=4600000, len=210, loss=1.799, n/ep=0, n/st=16, rew=6.67]                                                                 
Epoch #46: test_reward: 5.495500 ± 2.081885, best_reward: 5.606500 ± 1.929937 in #22
Epoch #47: 100001it [04:13, 393.97it/s, env_step=4700000, len=112, loss=1.777, n/ep=0, n/st=16, rew=4.55]                                                                 
Epoch #47: test_reward: 5.256500 ± 1.997535, best_reward: 5.606500 ± 1.929937 in #22
Epoch #48: 100001it [04:13, 394.40it/s, env_step=4800000, len=97, loss=1.817, n/ep=0, n/st=16, rew=2.78]                                                                  
Epoch #48: test_reward: 5.605500 ± 2.020414, best_reward: 5.606500 ± 1.929937 in #22
Epoch #49: 100001it [04:14, 392.94it/s, env_step=4900000, len=107, loss=1.737, n/ep=0, n/st=16, rew=6.75]                                                                 
Epoch #49: test_reward: 4.948100 ± 2.144773, best_reward: 5.606500 ± 1.929937 in #22
Epoch #50: 100001it [04:14, 393.08it/s, env_step=5000000, len=167, loss=1.692, n/ep=0, n/st=16, rew=9.67]                                                                 
Epoch #50: test_reward: 5.578400 ± 1.708149, best_reward: 5.606500 ± 1.929937 in #22
{'best_result': '5.61 ± 1.93',
 'best_reward': 5.606500000000002,
 'duration': '12952.98s',
 'test_episode': 5100,
 'test_speed': '2735.19 step/s',
 'test_step': 610556,
 'test_time': '223.22s',
 'train_episode': 39257,
 'train_speed': '392.78 step/s',
 'train_step': 5000000,
 'train_time/collector': '1926.29s',
 'train_time/model': '10803.47s'}

# DRQN

Run: 50_20251215_200440

uv run levd.py --algorithm drqn --scenario defend_the_center --train_levels 0 1 --train_maps 1 --test_levels 2 3 4 --test_maps 1 --seed 50 --epoch 50 --step-per-collect 10 --device cuda --batch-size 64 --buffer-size 200000

Epoch #1: 100001it [05:19, 313.32it/s, env_step=100000, len=95, loss=0.112, n/ep=0, n/st=16, rew=1.75]                                                                    
Epoch #1: test_reward: 2.836200 ± 2.374031, best_reward: 2.836200 ± 2.374031 in #1
Epoch #2: 100001it [05:17, 314.56it/s, env_step=200000, len=105, loss=0.136, n/ep=0, n/st=16, rew=2.68]                                                                   
Epoch #2: test_reward: 2.900800 ± 1.978173, best_reward: 2.900800 ± 1.978173 in #2
Epoch #3: 100001it [05:16, 316.17it/s, env_step=300000, len=97, loss=0.125, n/ep=0, n/st=16, rew=0.75]                                                                    
Epoch #3: test_reward: 3.202600 ± 1.924435, best_reward: 3.202600 ± 1.924435 in #3
Epoch #4: 100001it [05:16, 316.07it/s, env_step=400000, len=110, loss=0.127, n/ep=0, n/st=16, rew=3.58]                                                                   
Epoch #4: test_reward: 3.723400 ± 1.720052, best_reward: 3.723400 ± 1.720052 in #4
Epoch #5: 100001it [05:15, 316.65it/s, env_step=500000, len=141, loss=0.123, n/ep=0, n/st=16, rew=4.63]                                                                   
Epoch #5: test_reward: 3.066700 ± 2.092866, best_reward: 3.723400 ± 1.720052 in #4
Epoch #6: 100001it [05:14, 317.60it/s, env_step=600000, len=131, loss=0.109, n/ep=0, n/st=16, rew=3.67]                                                                   
Epoch #6: test_reward: 3.889900 ± 1.746986, best_reward: 3.889900 ± 1.746986 in #6
Epoch #7: 100001it [05:14, 317.50it/s, env_step=700000, len=70, loss=0.114, n/ep=0, n/st=16, rew=-0.15]                                                                   
Epoch #7: test_reward: 3.286600 ± 1.829679, best_reward: 3.889900 ± 1.746986 in #6
Epoch #8: 100001it [05:14, 317.77it/s, env_step=800000, len=81, loss=0.116, n/ep=0, n/st=16, rew=1.36]                                                                    
Epoch #8: test_reward: 3.411800 ± 2.052638, best_reward: 3.889900 ± 1.746986 in #6
Epoch #9: 100001it [05:14, 317.96it/s, env_step=900000, len=133, loss=0.134, n/ep=1, n/st=16, rew=4.66]                                                                   
Epoch #9: test_reward: 3.039200 ± 2.101015, best_reward: 3.889900 ± 1.746986 in #6
Epoch #10: 100001it [05:14, 318.09it/s, env_step=1000000, len=72, loss=0.113, n/ep=0, n/st=16, rew=0.89]                                                                  
Epoch #10: test_reward: 4.050700 ± 1.676560, best_reward: 4.050700 ± 1.676560 in #10
Epoch #11: 100001it [05:14, 317.92it/s, env_step=1100000, len=87, loss=0.102, n/ep=0, n/st=16, rew=1.84]                                                                  
Epoch #11: test_reward: 3.920200 ± 1.744848, best_reward: 4.050700 ± 1.676560 in #10
Epoch #12: 100001it [05:14, 317.85it/s, env_step=1200000, len=118, loss=0.095, n/ep=0, n/st=16, rew=4.69]                                                                 
Epoch #12: test_reward: 3.444300 ± 1.936017, best_reward: 4.050700 ± 1.676560 in #10
Epoch #13: 100001it [05:14, 317.78it/s, env_step=1300000, len=95, loss=0.098, n/ep=0, n/st=16, rew=2.71]                                                                  
Epoch #13: test_reward: 3.888200 ± 1.716560, best_reward: 4.050700 ± 1.676560 in #10
Epoch #14: 100001it [05:14, 317.96it/s, env_step=1400000, len=84, loss=0.086, n/ep=0, n/st=16, rew=2.79]                                                                  
Epoch #14: test_reward: 3.750200 ± 1.548019, best_reward: 4.050700 ± 1.676560 in #10
Epoch #15: 100001it [05:16, 316.20it/s, env_step=1500000, len=125, loss=0.091, n/ep=0, n/st=16, rew=5.53]                                                                 
Epoch #15: test_reward: 4.670600 ± 1.658233, best_reward: 4.670600 ± 1.658233 in #15
Epoch #16: 100001it [05:14, 317.63it/s, env_step=1600000, len=98, loss=0.085, n/ep=0, n/st=16, rew=4.54]                                                                  
Epoch #16: test_reward: 4.447700 ± 1.903664, best_reward: 4.670600 ± 1.658233 in #15
Epoch #17: 100001it [05:16, 316.05it/s, env_step=1700000, len=111, loss=0.091, n/ep=0, n/st=16, rew=4.69]                                                                 
Epoch #17: test_reward: 4.870900 ± 1.630299, best_reward: 4.870900 ± 1.630299 in #17
Epoch #18: 100001it [05:16, 316.06it/s, env_step=1800000, len=124, loss=0.088, n/ep=0, n/st=16, rew=5.70]                                                                 
Epoch #18: test_reward: 4.843400 ± 1.488117, best_reward: 4.870900 ± 1.630299 in #17
Epoch #19: 100001it [05:16, 315.54it/s, env_step=1900000, len=137, loss=0.093, n/ep=0, n/st=16, rew=4.62]                                                                 
Epoch #19: test_reward: 4.469500 ± 1.367927, best_reward: 4.870900 ± 1.630299 in #17
Epoch #20: 100001it [05:16, 315.97it/s, env_step=2000000, len=106, loss=0.085, n/ep=0, n/st=16, rew=3.48]                                                                 
Epoch #20: test_reward: 4.263000 ± 1.622524, best_reward: 4.870900 ± 1.630299 in #17
Epoch #21: 100001it [05:16, 316.34it/s, env_step=2100000, len=107, loss=0.093, n/ep=0, n/st=16, rew=3.69]                                                                 
Epoch #21: test_reward: 4.471600 ± 1.489133, best_reward: 4.870900 ± 1.630299 in #17
Epoch #22: 100001it [05:17, 315.27it/s, env_step=2200000, len=108, loss=0.080, n/ep=0, n/st=16, rew=4.69]                                                                 
Epoch #22: test_reward: 4.654300 ± 1.864406, best_reward: 4.870900 ± 1.630299 in #17
Epoch #23: 100001it [05:15, 316.69it/s, env_step=2300000, len=170, loss=0.081, n/ep=0, n/st=16, rew=6.43]                                                                 
Epoch #23: test_reward: 4.380000 ± 1.935233, best_reward: 4.870900 ± 1.630299 in #17
Epoch #24: 100001it [05:14, 317.87it/s, env_step=2400000, len=119, loss=0.072, n/ep=0, n/st=16, rew=2.80]                                                                 
Epoch #24: test_reward: 4.594300 ± 2.167671, best_reward: 4.870900 ± 1.630299 in #17
Epoch #25: 100001it [05:13, 319.41it/s, env_step=2500000, len=107, loss=0.079, n/ep=0, n/st=16, rew=3.86]                                                                 
Epoch #25: test_reward: 3.996400 ± 1.754717, best_reward: 4.870900 ± 1.630299 in #17
Epoch #26: 100001it [05:13, 318.71it/s, env_step=2600000, len=138, loss=0.072, n/ep=0, n/st=16, rew=5.67]                                                                 
Epoch #26: test_reward: 4.474400 ± 1.783351, best_reward: 4.870900 ± 1.630299 in #17
Epoch #27: 100001it [05:16, 315.53it/s, env_step=2700000, len=136, loss=0.072, n/ep=0, n/st=16, rew=5.69]                                                                 
Epoch #27: test_reward: 4.635900 ± 1.866251, best_reward: 4.870900 ± 1.630299 in #17
Epoch #28: 100001it [05:16, 315.47it/s, env_step=2800000, len=139, loss=0.073, n/ep=0, n/st=16, rew=5.64]                                                                 
Epoch #28: test_reward: 4.848100 ± 1.900556, best_reward: 4.870900 ± 1.630299 in #17
Epoch #29: 100001it [05:15, 316.68it/s, env_step=2900000, len=153, loss=0.083, n/ep=0, n/st=16, rew=6.70]                                                                 
Epoch #29: test_reward: 4.756300 ± 1.697557, best_reward: 4.870900 ± 1.630299 in #17
Epoch #30: 100001it [05:13, 318.89it/s, env_step=3000000, len=111, loss=0.077, n/ep=0, n/st=16, rew=2.66]                                                                 
Epoch #30: test_reward: 4.671100 ± 2.005622, best_reward: 4.870900 ± 1.630299 in #17
Epoch #31: 100001it [05:15, 317.12it/s, env_step=3100000, len=112, loss=0.088, n/ep=0, n/st=16, rew=3.69]                                                                 
Epoch #31: test_reward: 4.464500 ± 1.824065, best_reward: 4.870900 ± 1.630299 in #17
Epoch #32: 100001it [05:14, 317.52it/s, env_step=3200000, len=132, loss=0.080, n/ep=0, n/st=16, rew=5.68]                                                                 
Epoch #32: test_reward: 4.664800 ± 2.146992, best_reward: 4.870900 ± 1.630299 in #17
Epoch #33: 100001it [05:15, 317.07it/s, env_step=3300000, len=120, loss=0.081, n/ep=0, n/st=16, rew=4.67]                                                                 
Epoch #33: test_reward: 4.237700 ± 1.847785, best_reward: 4.870900 ± 1.630299 in #17
Epoch #34: 100001it [05:15, 317.05it/s, env_step=3400000, len=109, loss=0.075, n/ep=1, n/st=16, rew=2.76]                                                                 
Epoch #34: test_reward: 4.211800 ± 1.874876, best_reward: 4.870900 ± 1.630299 in #17
Epoch #35: 100001it [05:15, 317.01it/s, env_step=3500000, len=89, loss=0.073, n/ep=0, n/st=16, rew=2.77]                                                                  
Epoch #35: test_reward: 4.621700 ± 1.802896, best_reward: 4.870900 ± 1.630299 in #17
Epoch #36: 100001it [05:14, 317.56it/s, env_step=3600000, len=158, loss=0.071, n/ep=0, n/st=16, rew=5.63]                                                                 
Epoch #36: test_reward: 4.580000 ± 1.854658, best_reward: 4.870900 ± 1.630299 in #17
Epoch #37: 100001it [05:15, 316.60it/s, env_step=3700000, len=148, loss=0.079, n/ep=0, n/st=16, rew=6.41]                                                                 
Epoch #37: test_reward: 4.360700 ± 1.917444, best_reward: 4.870900 ± 1.630299 in #17
Epoch #38: 100001it [05:15, 317.03it/s, env_step=3800000, len=105, loss=0.070, n/ep=0, n/st=16, rew=3.75]                                                                 
Epoch #38: test_reward: 4.000700 ± 2.181411, best_reward: 4.870900 ± 1.630299 in #17
Epoch #39: 100001it [05:15, 317.28it/s, env_step=3900000, len=187, loss=0.087, n/ep=0, n/st=16, rew=6.59]                                                                 
Epoch #39: test_reward: 4.126700 ± 2.042592, best_reward: 4.870900 ± 1.630299 in #17
Epoch #40: 100001it [05:15, 316.77it/s, env_step=4000000, len=153, loss=0.081, n/ep=0, n/st=16, rew=5.50]                                                                 
Epoch #40: test_reward: 4.388400 ± 2.220784, best_reward: 4.870900 ± 1.630299 in #17
Epoch #41: 100001it [05:15, 317.40it/s, env_step=4100000, len=135, loss=0.078, n/ep=0, n/st=16, rew=4.66]                                                                 
Epoch #41: test_reward: 4.774200 ± 2.016069, best_reward: 4.870900 ± 1.630299 in #17
Epoch #42: 100001it [05:15, 317.05it/s, env_step=4200000, len=95, loss=0.073, n/ep=0, n/st=16, rew=1.71]                                                                  
Epoch #42: test_reward: 4.442400 ± 1.653744, best_reward: 4.870900 ± 1.630299 in #17
Epoch #43: 100001it [05:15, 317.01it/s, env_step=4300000, len=96, loss=0.076, n/ep=0, n/st=16, rew=4.52]                                                                  
Epoch #43: test_reward: 4.226200 ± 1.725651, best_reward: 4.870900 ± 1.630299 in #17
Epoch #44: 100001it [05:15, 316.62it/s, env_step=4400000, len=119, loss=0.069, n/ep=0, n/st=16, rew=5.71]                                                                 
Epoch #44: test_reward: 4.316600 ± 1.858429, best_reward: 4.870900 ± 1.630299 in #17
Epoch #45: 100001it [05:15, 316.48it/s, env_step=4500000, len=112, loss=0.077, n/ep=0, n/st=16, rew=5.70]                                                                 
Epoch #45: test_reward: 4.559600 ± 1.865674, best_reward: 4.870900 ± 1.630299 in #17
Epoch #46: 100001it [05:22, 310.36it/s, env_step=4600000, len=171, loss=0.065, n/ep=0, n/st=16, rew=6.45]                                                                 
Epoch #46: test_reward: 4.559600 ± 1.703352, best_reward: 4.870900 ± 1.630299 in #17
Epoch #47: 100001it [05:26, 306.25it/s, env_step=4700000, len=137, loss=0.074, n/ep=0, n/st=16, rew=6.66]                                                                 
Epoch #47: test_reward: 4.681000 ± 1.631921, best_reward: 4.870900 ± 1.630299 in #17
Epoch #48: 100001it [05:26, 306.45it/s, env_step=4800000, len=70, loss=0.068, n/ep=0, n/st=16, rew=1.82]                                                                  
Epoch #48: test_reward: 3.684000 ± 1.837914, best_reward: 4.870900 ± 1.630299 in #17
Epoch #49: 100001it [05:26, 306.69it/s, env_step=4900000, len=121, loss=0.070, n/ep=0, n/st=16, rew=4.69]                                                                 
Epoch #49: test_reward: 4.935700 ± 1.695405, best_reward: 4.935700 ± 1.695405 in #49
Epoch #50: 100001it [05:26, 306.61it/s, env_step=5000000, len=86, loss=0.071, n/ep=0, n/st=16, rew=1.82]                                                                  
Epoch #50: test_reward: 4.882900 ± 1.978321, best_reward: 4.935700 ± 1.695405 in #49
{'best_result': '4.94 ± 1.70',
 'best_reward': 4.935700000000003,
 'duration': '16045.94s',
 'test_episode': 5100,
 'test_speed': '2529.69 step/s',
 'test_step': 551070,
 'test_time': '217.84s',
 'train_episode': 44054,
 'train_speed': '315.89 step/s',
 'train_step': 5000000,
 'train_time/collector': '2004.50s',
 'train_time/model': '13823.60s'}

# DTQN

Run: 50_20251216_003720

uv run levd.py --algorithm dtqn --scenario defend_the_center --train_levels 0 1 --train_maps 1 --test_levels 2 3 4 --test_maps 1 --seed 50 --epoch 50 --step-per-collect 10 --device cuda --batch-size 64 --frame-stack 8

Epoch #1: 100001it [14:18, 116.42it/s, env_step=100000, len=105, loss=0.124, n/ep=0, n/st=16, rew=1.74]                                                                   
Epoch #1: test_reward: 4.944800 ± 1.472942, best_reward: 4.944800 ± 1.472942 in #1
Epoch #2: 100001it [13:56, 119.53it/s, env_step=200000, len=120, loss=0.132, n/ep=0, n/st=16, rew=2.67]                                                                   
Epoch #2: test_reward: 4.305600 ± 1.783063, best_reward: 4.944800 ± 1.472942 in #1
Epoch #3: 100001it [13:57, 119.34it/s, env_step=300000, len=93, loss=0.109, n/ep=0, n/st=16, rew=1.74]                                                                    
Epoch #3: test_reward: 5.284700 ± 1.670827, best_reward: 5.284700 ± 1.670827 in #3
Epoch #4: 100001it [13:56, 119.53it/s, env_step=400000, len=90, loss=0.128, n/ep=0, n/st=16, rew=1.75]                                                                    
Epoch #4: test_reward: 4.283900 ± 2.003492, best_reward: 5.284700 ± 1.670827 in #3
Epoch #5: 100001it [13:55, 119.75it/s, env_step=500000, len=94, loss=0.139, n/ep=0, n/st=16, rew=2.76]                                                                    
Epoch #5: test_reward: 4.584400 ± 2.754403, best_reward: 5.284700 ± 1.670827 in #3
Epoch #6: 100001it [13:55, 119.64it/s, env_step=600000, len=102, loss=0.132, n/ep=0, n/st=16, rew=2.70]                                                                   
Epoch #6: test_reward: 5.126100 ± 1.646374, best_reward: 5.284700 ± 1.670827 in #3
Epoch #7: 100001it [13:58, 119.30it/s, env_step=700000, len=129, loss=0.128, n/ep=1, n/st=16, rew=6.68]                                                                   
Epoch #7: test_reward: 5.149600 ± 1.766806, best_reward: 5.284700 ± 1.670827 in #3
Epoch #8: 100001it [13:59, 119.17it/s, env_step=800000, len=134, loss=0.140, n/ep=0, n/st=16, rew=5.70]                                                                   
Epoch #8: test_reward: 5.199700 ± 1.973086, best_reward: 5.284700 ± 1.670827 in #3
Epoch #9: 100001it [13:58, 119.26it/s, env_step=900000, len=187, loss=0.144, n/ep=0, n/st=16, rew=8.47]                                                                   
Epoch #9: test_reward: 5.732300 ± 1.855721, best_reward: 5.732300 ± 1.855721 in #9
Epoch #10: 100001it [13:58, 119.19it/s, env_step=1000000, len=133, loss=0.144, n/ep=0, n/st=16, rew=5.73]                                                                 
Epoch #10: test_reward: 5.120800 ± 1.528074, best_reward: 5.732300 ± 1.855721 in #9
Epoch #11: 100001it [13:59, 119.18it/s, env_step=1100000, len=100, loss=0.131, n/ep=0, n/st=16, rew=4.77]                                                                 
Epoch #11: test_reward: 3.867100 ± 2.383302, best_reward: 5.732300 ± 1.855721 in #9
Epoch #12: 100001it [13:59, 119.14it/s, env_step=1200000, len=159, loss=0.134, n/ep=1, n/st=16, rew=7.66]                                                                 
Epoch #12: test_reward: 5.283800 ± 2.540681, best_reward: 5.732300 ± 1.855721 in #9
Epoch #13: 100001it [13:59, 119.19it/s, env_step=1300000, len=129, loss=0.132, n/ep=2, n/st=16, rew=6.16]                                                                 
Epoch #13: test_reward: 5.443300 ± 1.688213, best_reward: 5.732300 ± 1.855721 in #9
Epoch #14: 100001it [13:59, 119.15it/s, env_step=1400000, len=120, loss=0.136, n/ep=0, n/st=16, rew=5.74]                                                                 
Epoch #14: test_reward: 4.820600 ± 2.099366, best_reward: 5.732300 ± 1.855721 in #9
Epoch #15: 100001it [14:00, 119.02it/s, env_step=1500000, len=131, loss=0.139, n/ep=0, n/st=16, rew=4.66]                                                                 
Epoch #15: test_reward: 5.143300 ± 1.670690, best_reward: 5.732300 ± 1.855721 in #9
Epoch #16: 100001it [13:59, 119.13it/s, env_step=1600000, len=129, loss=0.132, n/ep=0, n/st=16, rew=5.68]                                                                 
Epoch #16: test_reward: 4.881100 ± 2.263196, best_reward: 5.732300 ± 1.855721 in #9
Epoch #17: 100001it [13:58, 119.27it/s, env_step=1700000, len=144, loss=0.131, n/ep=0, n/st=16, rew=4.64]                                                                 
Epoch #17: test_reward: 5.263300 ± 1.806122, best_reward: 5.732300 ± 1.855721 in #9
Epoch #18: 100001it [13:59, 119.06it/s, env_step=1800000, len=145, loss=0.129, n/ep=0, n/st=16, rew=5.67]                                                                 
Epoch #18: test_reward: 5.013800 ± 1.836553, best_reward: 5.732300 ± 1.855721 in #9
Epoch #19: 100001it [13:59, 119.06it/s, env_step=1900000, len=66, loss=0.141, n/ep=0, n/st=16, rew=2.80]                                                                  
Epoch #19: test_reward: 5.318300 ± 2.130149, best_reward: 5.732300 ± 1.855721 in #9
Epoch #20: 100001it [14:00, 118.96it/s, env_step=2000000, len=136, loss=0.125, n/ep=0, n/st=16, rew=5.70]                                                                 
Epoch #20: test_reward: 5.467200 ± 2.138944, best_reward: 5.732300 ± 1.855721 in #9
Epoch #21: 100001it [14:00, 119.03it/s, env_step=2100000, len=128, loss=0.123, n/ep=0, n/st=16, rew=6.65]                                                                 
Epoch #21: test_reward: 4.518900 ± 1.703901, best_reward: 5.732300 ± 1.855721 in #9
Epoch #22: 100001it [13:59, 119.06it/s, env_step=2200000, len=113, loss=0.122, n/ep=1, n/st=16, rew=5.70]                                                                 
Epoch #22: test_reward: 3.752600 ± 2.697848, best_reward: 5.732300 ± 1.855721 in #9
Epoch #23: 100001it [14:00, 118.94it/s, env_step=2300000, len=148, loss=0.131, n/ep=1, n/st=16, rew=5.67]                                                                 
Epoch #23: test_reward: 4.430400 ± 1.845890, best_reward: 5.732300 ± 1.855721 in #9
Epoch #24: 100001it [14:00, 118.99it/s, env_step=2400000, len=92, loss=0.120, n/ep=0, n/st=16, rew=2.68]                                                                  
Epoch #24: test_reward: 5.017400 ± 1.851879, best_reward: 5.732300 ± 1.855721 in #9
Epoch #25: 100001it [14:00, 119.05it/s, env_step=2500000, len=150, loss=0.132, n/ep=0, n/st=16, rew=6.65]                                                                 
Epoch #25: test_reward: 5.046400 ± 1.908935, best_reward: 5.732300 ± 1.855721 in #9
Epoch #26: 100001it [14:00, 118.94it/s, env_step=2600000, len=131, loss=0.118, n/ep=0, n/st=16, rew=4.67]                                                                 
Epoch #26: test_reward: 4.803200 ± 2.240659, best_reward: 5.732300 ± 1.855721 in #9
Epoch #27: 100001it [14:00, 118.92it/s, env_step=2700000, len=101, loss=0.119, n/ep=0, n/st=16, rew=2.75]                                                                 
Epoch #27: test_reward: 5.293900 ± 1.636123, best_reward: 5.732300 ± 1.855721 in #9
Epoch #28: 100001it [14:00, 118.91it/s, env_step=2800000, len=111, loss=0.109, n/ep=1, n/st=16, rew=4.65]                                                                 
Epoch #28: test_reward: 4.763100 ± 1.678006, best_reward: 5.732300 ± 1.855721 in #9
Epoch #29: 100001it [14:01, 118.87it/s, env_step=2900000, len=165, loss=0.117, n/ep=0, n/st=16, rew=6.56]                                                                 
Epoch #29: test_reward: 4.274100 ± 1.745907, best_reward: 5.732300 ± 1.855721 in #9
Epoch #30: 100001it [14:00, 118.98it/s, env_step=3000000, len=111, loss=0.115, n/ep=0, n/st=16, rew=3.68]                                                                 
Epoch #30: test_reward: 5.314400 ± 2.355703, best_reward: 5.732300 ± 1.855721 in #9
Epoch #31: 100001it [14:00, 118.96it/s, env_step=3100000, len=66, loss=0.118, n/ep=0, n/st=16, rew=1.82]                                                                  
Epoch #31: test_reward: 4.831700 ± 1.767268, best_reward: 5.732300 ± 1.855721 in #9
Epoch #32: 100001it [14:00, 118.94it/s, env_step=3200000, len=79, loss=0.113, n/ep=0, n/st=16, rew=0.82]                                                                  
Epoch #32: test_reward: 5.284900 ± 2.354333, best_reward: 5.732300 ± 1.855721 in #9
Epoch #33: 100001it [14:01, 118.85it/s, env_step=3300000, len=148, loss=0.119, n/ep=0, n/st=16, rew=6.67]                                                                 
Epoch #33: test_reward: 5.016800 ± 1.749454, best_reward: 5.732300 ± 1.855721 in #9
Epoch #34: 100001it [14:01, 118.83it/s, env_step=3400000, len=127, loss=0.120, n/ep=0, n/st=16, rew=4.66]                                                                 
Epoch #34: test_reward: 4.775000 ± 2.105922, best_reward: 5.732300 ± 1.855721 in #9
Epoch #35: 100001it [14:02, 118.70it/s, env_step=3500000, len=94, loss=0.131, n/ep=0, n/st=16, rew=4.59]                                                                  
Epoch #35: test_reward: 3.649100 ± 2.290546, best_reward: 5.732300 ± 1.855721 in #9
Epoch #36: 100001it [14:00, 118.94it/s, env_step=3600000, len=129, loss=0.136, n/ep=0, n/st=16, rew=5.75]                                                                 
Epoch #36: test_reward: 3.842900 ± 2.123137, best_reward: 5.732300 ± 1.855721 in #9
Epoch #37: 100001it [14:01, 118.89it/s, env_step=3700000, len=152, loss=0.122, n/ep=0, n/st=16, rew=5.71]                                                                 
Epoch #37: test_reward: 4.360500 ± 2.402428, best_reward: 5.732300 ± 1.855721 in #9
Epoch #38: 100001it [14:04, 118.46it/s, env_step=3800000, len=159, loss=0.122, n/ep=0, n/st=16, rew=6.67]                                                                 
Epoch #38: test_reward: 5.038800 ± 1.826121, best_reward: 5.732300 ± 1.855721 in #9
Epoch #39: 100001it [14:02, 118.64it/s, env_step=3900000, len=111, loss=0.133, n/ep=0, n/st=16, rew=3.73]                                                                 
Epoch #39: test_reward: 4.673800 ± 2.318322, best_reward: 5.732300 ± 1.855721 in #9
Epoch #40: 100001it [14:11, 117.46it/s, env_step=4000000, len=147, loss=0.130, n/ep=0, n/st=16, rew=6.69]                                                                 
Epoch #40: test_reward: 4.709000 ± 2.200852, best_reward: 5.732300 ± 1.855721 in #9
Epoch #41: 100001it [14:14, 117.04it/s, env_step=4100000, len=122, loss=0.134, n/ep=1, n/st=16, rew=6.45]                                                                 
Epoch #41: test_reward: 4.829500 ± 2.105915, best_reward: 5.732300 ± 1.855721 in #9
Epoch #42: 100001it [14:05, 118.33it/s, env_step=4200000, len=160, loss=0.127, n/ep=0, n/st=16, rew=6.54]                                                                 
Epoch #42: test_reward: 4.910900 ± 1.823288, best_reward: 5.732300 ± 1.855721 in #9
Epoch #43: 100001it [14:02, 118.64it/s, env_step=4300000, len=140, loss=0.122, n/ep=0, n/st=16, rew=5.68]                                                                 
Epoch #43: test_reward: 4.913200 ± 1.524212, best_reward: 5.732300 ± 1.855721 in #9
Epoch #44: 100001it [14:03, 118.56it/s, env_step=4400000, len=152, loss=0.125, n/ep=1, n/st=16, rew=8.49]                                                                 
Epoch #44: test_reward: 4.631000 ± 1.628753, best_reward: 5.732300 ± 1.855721 in #9
Epoch #45: 100001it [14:03, 118.57it/s, env_step=4500000, len=155, loss=0.113, n/ep=0, n/st=16, rew=5.71]                                                                 
Epoch #45: test_reward: 5.198800 ± 1.696786, best_reward: 5.732300 ± 1.855721 in #9
Epoch #46: 100001it [14:02, 118.71it/s, env_step=4600000, len=193, loss=0.119, n/ep=0, n/st=16, rew=7.66]                                                                 
Epoch #46: test_reward: 4.954600 ± 1.779352, best_reward: 5.732300 ± 1.855721 in #9
Epoch #47: 100001it [14:09, 117.77it/s, env_step=4700000, len=144, loss=0.122, n/ep=0, n/st=16, rew=5.65]                                                                 
Epoch #47: test_reward: 4.739600 ± 1.826552, best_reward: 5.732300 ± 1.855721 in #9
Epoch #48: 100001it [14:17, 116.69it/s, env_step=4800000, len=93, loss=0.116, n/ep=0, n/st=16, rew=3.79]                                                                  
Epoch #48: test_reward: 4.186000 ± 1.555224, best_reward: 5.732300 ± 1.855721 in #9
Epoch #49: 100001it [14:07, 117.98it/s, env_step=4900000, len=123, loss=0.116, n/ep=0, n/st=16, rew=6.67]                                                                 
Epoch #49: test_reward: 4.547900 ± 1.804159, best_reward: 5.732300 ± 1.855721 in #9
Epoch #50: 100001it [14:11, 117.50it/s, env_step=5000000, len=128, loss=0.121, n/ep=0, n/st=16, rew=4.45]                                                                 
Epoch #50: test_reward: 5.151700 ± 1.952629, best_reward: 5.732300 ± 1.855721 in #9
{'best_result': '5.73 ± 1.86',
 'best_reward': 5.732300000000002,
 'duration': '42406.55s',
 'test_episode': 5100,
 'test_speed': '1926.53 step/s',
 'test_step': 588071,
 'test_time': '305.25s',
 'train_episode': 40627,
 'train_speed': '118.76 step/s',
 'train_step': 5000000,
 'train_time/collector': '2599.35s',
 'train_time/model': '39501.95s'}
