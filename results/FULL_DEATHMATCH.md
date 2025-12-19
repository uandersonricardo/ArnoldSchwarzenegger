# RAINBOW

Run: 1_20251211_225841

uv run levd.py --algorithm rainbow --scenario full_deathmatch --train_levels 0 --train_maps 1 14 --test_levels 0 --test_maps 4 7 --seed 1 --epoch 80 --step-per-collect 10 --device cuda --batch-size 64

Epoch #1: 100001it [05:49, 286.34it/s, env_step=100000, len=32, loss=2.485, n/ep=0, n/st=10, rew=-0.21]                                                                                                                                                   
Epoch #1: test_reward: 4.730406 ± 12.213516, best_reward: 6.345074 ± 14.397328 in #0
Epoch #2: 100001it [05:48, 287.00it/s, env_step=200000, len=74, loss=2.463, n/ep=1, n/st=10, rew=11.68]                                                                                                                                                   
Epoch #2: test_reward: 8.486800 ± 15.958512, best_reward: 8.486800 ± 15.958512 in #2
Epoch #3: 100001it [05:51, 284.37it/s, env_step=300000, len=18, loss=2.285, n/ep=0, n/st=10, rew=-0.12]                                                                                                                                                   
Epoch #3: test_reward: 5.792748 ± 12.892487, best_reward: 8.486800 ± 15.958512 in #2
Epoch #4: 100001it [05:50, 285.65it/s, env_step=400000, len=16, loss=2.149, n/ep=0, n/st=10, rew=-0.22]                                                                                                                                                   
Epoch #4: test_reward: 4.538098 ± 10.519203, best_reward: 8.486800 ± 15.958512 in #2
Epoch #5: 100001it [05:47, 287.84it/s, env_step=500000, len=42, loss=2.183, n/ep=0, n/st=10, rew=-0.25]                                                                                                                                                   
Epoch #5: test_reward: 7.432229 ± 14.432064, best_reward: 8.486800 ± 15.958512 in #2
Epoch #6: 100001it [05:48, 286.78it/s, env_step=600000, len=47, loss=1.945, n/ep=1, n/st=10, rew=-0.65]                                                                                                                                                   
Epoch #6: test_reward: 5.743389 ± 12.849289, best_reward: 8.486800 ± 15.958512 in #2
Epoch #7: 100001it [05:50, 285.11it/s, env_step=700000, len=39, loss=2.012, n/ep=0, n/st=10, rew=-0.23]                                                                                                                                                   
Epoch #7: test_reward: 2.457167 ± 7.132384, best_reward: 8.486800 ± 15.958512 in #2
Epoch #8: 100001it [05:47, 287.55it/s, env_step=800000, len=18, loss=1.908, n/ep=0, n/st=10, rew=1.41]                                                                                                                                                    
Epoch #8: test_reward: 5.050633 ± 10.344327, best_reward: 8.486800 ± 15.958512 in #2
Epoch #9: 100001it [05:51, 284.77it/s, env_step=900000, len=29, loss=1.846, n/ep=0, n/st=10, rew=4.21]                                                                                                                                                    
Epoch #9: test_reward: 3.017989 ± 7.572989, best_reward: 8.486800 ± 15.958512 in #2
Epoch #10: 100001it [05:51, 284.14it/s, env_step=1000000, len=12, loss=1.799, n/ep=0, n/st=10, rew=-0.08]                                                                                                                                                 
Epoch #10: test_reward: 3.696999 ± 9.377963, best_reward: 8.486800 ± 15.958512 in #2
Epoch #11: 100001it [05:45, 289.59it/s, env_step=1100000, len=1, loss=1.749, n/ep=0, n/st=10, rew=0.00]                                                                                                                                                   
Epoch #11: test_reward: 4.285801 ± 8.120635, best_reward: 8.486800 ± 15.958512 in #2
Epoch #12: 100001it [05:50, 285.59it/s, env_step=1200000, len=30, loss=1.670, n/ep=1, n/st=10, rew=3.05]                                                                                                                                                  
Epoch #12: test_reward: 4.093671 ± 8.099605, best_reward: 8.486800 ± 15.958512 in #2
Epoch #13: 100001it [05:46, 288.65it/s, env_step=1300000, len=29, loss=1.698, n/ep=0, n/st=10, rew=0.00]                                                                                                                                                  
Epoch #13: test_reward: 3.400155 ± 8.057260, best_reward: 8.486800 ± 15.958512 in #2
Epoch #14: 100001it [05:47, 288.17it/s, env_step=1400000, len=24, loss=1.650, n/ep=0, n/st=10, rew=2.63]                                                                                                                                                  
Epoch #14: test_reward: 3.218847 ± 8.031833, best_reward: 8.486800 ± 15.958512 in #2
Epoch #15: 100001it [05:46, 288.87it/s, env_step=1500000, len=89, loss=1.654, n/ep=0, n/st=10, rew=-0.01]                                                                                                                                                 
Epoch #15: test_reward: 3.232814 ± 7.962565, best_reward: 8.486800 ± 15.958512 in #2
Epoch #16: 100001it [05:49, 285.95it/s, env_step=1600000, len=24, loss=1.557, n/ep=0, n/st=10, rew=0.04]                                                                                                                                                  
Epoch #16: test_reward: 4.337929 ± 8.648420, best_reward: 8.486800 ± 15.958512 in #2
Epoch #17: 100001it [05:49, 286.26it/s, env_step=1700000, len=50, loss=1.479, n/ep=0, n/st=10, rew=-0.00]                                                                                                                                                 
Epoch #17: test_reward: 3.583957 ± 8.420894, best_reward: 8.486800 ± 15.958512 in #2
Epoch #18: 100001it [05:44, 290.04it/s, env_step=1800000, len=1, loss=1.536, n/ep=2, n/st=10, rew=0.00]                                                                                                                                                   
Epoch #18: test_reward: 6.529005 ± 13.440642, best_reward: 8.486800 ± 15.958512 in #2
Epoch #19: 100001it [05:46, 288.78it/s, env_step=1900000, len=40, loss=1.473, n/ep=0, n/st=10, rew=-0.11]                                                                                                                                                 
Epoch #19: test_reward: 2.799097 ± 7.139287, best_reward: 8.486800 ± 15.958512 in #2
Epoch #20: 100001it [05:43, 290.74it/s, env_step=2000000, len=37, loss=1.445, n/ep=0, n/st=10, rew=6.92]                                                                                                                                                  
Epoch #20: test_reward: 4.222154 ± 11.130101, best_reward: 8.486800 ± 15.958512 in #2
Epoch #21: 100001it [05:45, 289.33it/s, env_step=2100000, len=13, loss=1.401, n/ep=0, n/st=10, rew=-0.00]                                                                                                                                                 
Epoch #21: test_reward: 6.109123 ± 11.433831, best_reward: 8.486800 ± 15.958512 in #2
Epoch #22: 100001it [05:42, 291.67it/s, env_step=2200000, len=154, loss=1.314, n/ep=0, n/st=10, rew=-0.02]                                                                                                                                                
Epoch #22: test_reward: 5.682101 ± 13.287268, best_reward: 8.486800 ± 15.958512 in #2
Epoch #23: 100001it [05:45, 289.39it/s, env_step=2300000, len=1, loss=1.277, n/ep=0, n/st=10, rew=0.00]                                                                                                                                                   
Epoch #23: test_reward: 5.536094 ± 12.994058, best_reward: 8.486800 ± 15.958512 in #2
Epoch #24: 100001it [05:43, 290.92it/s, env_step=2400000, len=32, loss=1.269, n/ep=0, n/st=10, rew=-0.21]                                                                                                                                                 
Epoch #24: test_reward: 4.609906 ± 11.371071, best_reward: 8.486800 ± 15.958512 in #2
Epoch #25: 100001it [05:46, 288.39it/s, env_step=2500000, len=1, loss=1.242, n/ep=0, n/st=10, rew=0.00]                                                                                                                                                   
Epoch #25: test_reward: 4.620941 ± 10.991561, best_reward: 8.486800 ± 15.958512 in #2
Epoch #26: 100001it [05:47, 287.74it/s, env_step=2600000, len=1, loss=1.225, n/ep=0, n/st=10, rew=0.00]                                                                                                                                                   
Epoch #26: test_reward: 6.179617 ± 13.200020, best_reward: 8.486800 ± 15.958512 in #2
Epoch #27: 100001it [05:47, 287.67it/s, env_step=2700000, len=1, loss=1.207, n/ep=1, n/st=10, rew=0.00]                                                                                                                                                   
Epoch #27: test_reward: 4.646164 ± 9.810736, best_reward: 8.486800 ± 15.958512 in #2
Epoch #28: 100001it [05:48, 286.66it/s, env_step=2800000, len=1, loss=1.204, n/ep=0, n/st=10, rew=0.00]                                                                                                                                                   
Epoch #28: test_reward: 4.890884 ± 11.454582, best_reward: 8.486800 ± 15.958512 in #2
Epoch #29: 100001it [05:46, 288.26it/s, env_step=2900000, len=1, loss=1.123, n/ep=0, n/st=10, rew=0.00]                                                                                                                                                   
Epoch #29: test_reward: 6.324511 ± 12.884518, best_reward: 8.486800 ± 15.958512 in #2
Epoch #30: 100001it [05:44, 290.53it/s, env_step=3000000, len=54, loss=1.104, n/ep=1, n/st=10, rew=-0.22]                                                                                                                                                 
Epoch #30: test_reward: 5.179299 ± 12.109939, best_reward: 8.486800 ± 15.958512 in #2
Epoch #31: 100001it [05:45, 289.65it/s, env_step=3100000, len=66, loss=1.097, n/ep=0, n/st=10, rew=-0.02]                                                                                                                                                 
Epoch #31: test_reward: 4.179562 ± 9.187080, best_reward: 8.486800 ± 15.958512 in #2
Epoch #32: 100001it [05:49, 285.98it/s, env_step=3200000, len=30, loss=1.080, n/ep=0, n/st=10, rew=0.01]                                                                                                                                                  
Epoch #32: test_reward: 4.097503 ± 10.223413, best_reward: 8.486800 ± 15.958512 in #2
Epoch #33: 100001it [05:49, 286.43it/s, env_step=3300000, len=82, loss=1.063, n/ep=1, n/st=10, rew=-0.40]                                                                                                                                                 
Epoch #33: test_reward: 3.819948 ± 7.906024, best_reward: 8.486800 ± 15.958512 in #2
Epoch #34: 100001it [05:45, 289.73it/s, env_step=3400000, len=36, loss=1.039, n/ep=3, n/st=10, rew=5.05]                                                                                                                                                  
Epoch #34: test_reward: 3.592340 ± 9.995828, best_reward: 8.486800 ± 15.958512 in #2
Epoch #35: 100001it [05:52, 283.67it/s, env_step=3500000, len=25, loss=1.049, n/ep=0, n/st=10, rew=0.02]                                                                                                                                                  
Epoch #35: test_reward: 3.774748 ± 8.632123, best_reward: 8.486800 ± 15.958512 in #2
Epoch #36: 100001it [05:48, 286.68it/s, env_step=3600000, len=89, loss=1.052, n/ep=0, n/st=10, rew=-0.18]                                                                                                                                                 
Epoch #36: test_reward: 3.799535 ± 8.459274, best_reward: 8.486800 ± 15.958512 in #2
Epoch #37: 100001it [05:48, 287.28it/s, env_step=3700000, len=43, loss=1.047, n/ep=0, n/st=10, rew=0.01]                                                                                                                                                  
Epoch #37: test_reward: 6.053231 ± 12.383924, best_reward: 8.486800 ± 15.958512 in #2
Epoch #38: 100001it [05:47, 287.43it/s, env_step=3800000, len=54, loss=1.036, n/ep=0, n/st=10, rew=7.01]                                                                                                                                                  
Epoch #38: test_reward: 6.213268 ± 13.305044, best_reward: 8.486800 ± 15.958512 in #2
Epoch #39: 100001it [05:45, 289.72it/s, env_step=3900000, len=97, loss=1.032, n/ep=0, n/st=10, rew=11.78]                                                                                                                                                 
Epoch #39: test_reward: 5.136206 ± 10.702057, best_reward: 8.486800 ± 15.958512 in #2
Epoch #40: 100001it [05:47, 287.67it/s, env_step=4000000, len=16, loss=0.984, n/ep=0, n/st=10, rew=-0.11]                                                                                                                                                 
Epoch #40: test_reward: 3.147278 ± 8.793812, best_reward: 8.486800 ± 15.958512 in #2
Epoch #41: 100001it [05:49, 285.88it/s, env_step=4100000, len=14, loss=1.035, n/ep=0, n/st=10, rew=0.01]                                                                                                                                                  
Epoch #41: test_reward: 4.060933 ± 8.961741, best_reward: 8.486800 ± 15.958512 in #2
Epoch #42: 100001it [05:47, 287.61it/s, env_step=4200000, len=12, loss=1.003, n/ep=0, n/st=10, rew=-0.00]                                                                                                                                                 
Epoch #42: test_reward: 4.054354 ± 9.632257, best_reward: 8.486800 ± 15.958512 in #2
Epoch #43: 100001it [05:48, 287.26it/s, env_step=4300000, len=31, loss=1.009, n/ep=0, n/st=10, rew=0.00]                                                                                                                                                  
Epoch #43: test_reward: 2.899550 ± 7.495513, best_reward: 8.486800 ± 15.958512 in #2
Epoch #44: 100001it [05:50, 285.62it/s, env_step=4400000, len=32, loss=1.003, n/ep=0, n/st=10, rew=-0.20]                                                                                                                                                 
Epoch #44: test_reward: 5.531520 ± 10.025525, best_reward: 8.486800 ± 15.958512 in #2
Epoch #45: 100001it [05:48, 286.87it/s, env_step=4500000, len=35, loss=1.010, n/ep=0, n/st=10, rew=0.04]                                                                                                                                                  
Epoch #45: test_reward: 4.884745 ± 8.733519, best_reward: 8.486800 ± 15.958512 in #2
Epoch #46: 100001it [05:45, 289.69it/s, env_step=4600000, len=1, loss=1.035, n/ep=0, n/st=10, rew=0.00]                                                                                                                                                   
Epoch #46: test_reward: 4.892676 ± 11.869064, best_reward: 8.486800 ± 15.958512 in #2
Epoch #47: 100001it [05:48, 287.10it/s, env_step=4700000, len=27, loss=0.960, n/ep=0, n/st=10, rew=0.01]                                                                                                                                                  
Epoch #47: test_reward: 5.346428 ± 12.247334, best_reward: 8.486800 ± 15.958512 in #2
Epoch #48: 100001it [05:48, 287.23it/s, env_step=4800000, len=28, loss=1.029, n/ep=0, n/st=10, rew=0.02]                                                                                                                                                  
Epoch #48: test_reward: 4.356476 ± 9.814746, best_reward: 8.486800 ± 15.958512 in #2
Epoch #49: 100001it [05:48, 286.74it/s, env_step=4900000, len=59, loss=0.990, n/ep=0, n/st=10, rew=-0.39]                                                                                                                                                 
Epoch #49: test_reward: 4.508352 ± 10.041912, best_reward: 8.486800 ± 15.958512 in #2
Epoch #50: 100001it [05:49, 286.08it/s, env_step=5000000, len=34, loss=0.980, n/ep=0, n/st=10, rew=-0.16]                                                                                                                                                 
Epoch #50: test_reward: 5.839703 ± 11.275479, best_reward: 8.486800 ± 15.958512 in #2
Epoch #51: 100001it [05:41, 293.12it/s, env_step=5100000, len=16, loss=1.035, n/ep=0, n/st=10, rew=-0.08]                                                                                                                                                 
Epoch #51: test_reward: 5.547409 ± 11.310083, best_reward: 8.486800 ± 15.958512 in #2
Epoch #52: 100001it [05:49, 286.18it/s, env_step=5200000, len=41, loss=1.043, n/ep=1, n/st=10, rew=-0.17]                                                                                                                                                 
Epoch #52: test_reward: 3.950486 ± 9.198729, best_reward: 8.486800 ± 15.958512 in #2
Epoch #53: 100001it [05:49, 286.04it/s, env_step=5300000, len=1, loss=1.031, n/ep=0, n/st=10, rew=0.00]                                                                                                                                                   
Epoch #53: test_reward: 6.459999 ± 13.031250, best_reward: 8.486800 ± 15.958512 in #2
Epoch #54: 100001it [05:47, 287.47it/s, env_step=5400000, len=25, loss=1.029, n/ep=1, n/st=10, rew=0.01]                                                                                                                                                  
Epoch #54: test_reward: 5.359643 ± 12.791898, best_reward: 8.486800 ± 15.958512 in #2
Epoch #55: 100001it [05:51, 284.86it/s, env_step=5500000, len=47, loss=0.997, n/ep=0, n/st=10, rew=-0.19]                                                                                                                                                 
Epoch #55: test_reward: 7.029847 ± 12.277644, best_reward: 8.486800 ± 15.958512 in #2
Epoch #56: 100001it [05:44, 290.42it/s, env_step=5600000, len=29, loss=1.009, n/ep=0, n/st=10, rew=-0.01]                                                                                                                                                 
Epoch #56: test_reward: 7.496965 ± 12.717089, best_reward: 8.486800 ± 15.958512 in #2
Epoch #57: 100001it [05:52, 283.75it/s, env_step=5700000, len=14, loss=1.047, n/ep=0, n/st=10, rew=-0.06]                                                                                                                                                 
Epoch #57: test_reward: 4.242885 ± 11.204486, best_reward: 8.486800 ± 15.958512 in #2
Epoch #58: 100001it [05:50, 285.20it/s, env_step=5800000, len=1, loss=1.073, n/ep=3, n/st=10, rew=0.00]                                                                                                                                                   
Epoch #58: test_reward: 5.181342 ± 10.893888, best_reward: 8.486800 ± 15.958512 in #2
Epoch #59: 100001it [05:52, 284.02it/s, env_step=5900000, len=51, loss=1.017, n/ep=0, n/st=10, rew=-0.18]                                                                                                                                                 
Epoch #59: test_reward: 3.384254 ± 8.574850, best_reward: 8.486800 ± 15.958512 in #2
Epoch #60: 100001it [05:51, 284.26it/s, env_step=6000000, len=20, loss=1.000, n/ep=0, n/st=10, rew=-0.11]                                                                                                                                                 
Epoch #60: test_reward: 5.074110 ± 9.806845, best_reward: 8.486800 ± 15.958512 in #2
Epoch #61: 100001it [05:49, 286.47it/s, env_step=6100000, len=1, loss=1.030, n/ep=1, n/st=10, rew=0.00]                                                                                                                                                   
Epoch #61: test_reward: 3.228635 ± 8.292948, best_reward: 8.486800 ± 15.958512 in #2
Epoch #62: 100001it [05:42, 292.39it/s, env_step=6200000, len=28, loss=1.028, n/ep=1, n/st=10, rew=0.00]                                                                                                                                                  
Epoch #62: test_reward: 3.873837 ± 9.048955, best_reward: 8.486800 ± 15.958512 in #2
Epoch #63: 100001it [05:42, 292.15it/s, env_step=6300000, len=26, loss=1.037, n/ep=1, n/st=10, rew=0.00]                                                                                                                                                  
Epoch #63: test_reward: 3.335674 ± 11.691342, best_reward: 8.486800 ± 15.958512 in #2
Epoch #64: 100001it [05:45, 289.36it/s, env_step=6400000, len=30, loss=1.016, n/ep=0, n/st=10, rew=-0.40]                                                                                                                                                 
Epoch #64: test_reward: 3.613721 ± 8.193723, best_reward: 8.486800 ± 15.958512 in #2
Epoch #65: 100001it [05:47, 287.75it/s, env_step=6500000, len=33, loss=1.024, n/ep=1, n/st=10, rew=-0.19]                                                                                                                                                 
Epoch #65: test_reward: 4.123538 ± 8.584179, best_reward: 8.486800 ± 15.958512 in #2
Epoch #66: 100001it [05:52, 283.77it/s, env_step=6600000, len=34, loss=1.002, n/ep=0, n/st=10, rew=-0.19]                                                                                                                                                 
Epoch #66: test_reward: 4.777649 ± 10.188818, best_reward: 8.486800 ± 15.958512 in #2
Epoch #67: 100001it [05:46, 288.96it/s, env_step=6700000, len=32, loss=1.034, n/ep=0, n/st=10, rew=-0.16]                                                                                                                                                 
Epoch #67: test_reward: 2.103741 ± 6.075867, best_reward: 8.486800 ± 15.958512 in #2
Epoch #68: 100001it [05:47, 287.53it/s, env_step=6800000, len=28, loss=1.053, n/ep=0, n/st=10, rew=0.00]                                                                                                                                                  
Epoch #68: test_reward: 3.682362 ± 11.150791, best_reward: 8.486800 ± 15.958512 in #2
Epoch #69: 100001it [05:51, 284.32it/s, env_step=6900000, len=89, loss=1.058, n/ep=1, n/st=10, rew=16.39]                                                                                                                                                 
Epoch #69: test_reward: 3.212390 ± 8.008758, best_reward: 8.486800 ± 15.958512 in #2
Epoch #70: 100001it [05:46, 288.50it/s, env_step=7000000, len=1, loss=1.025, n/ep=1, n/st=10, rew=0.00]                                                                                                                                                   
Epoch #70: test_reward: 3.314058 ± 10.028005, best_reward: 8.486800 ± 15.958512 in #2
Epoch #71: 100001it [05:50, 285.06it/s, env_step=7100000, len=1, loss=1.044, n/ep=1, n/st=10, rew=0.00]                                                                                                                                                   
Epoch #71: test_reward: 3.886485 ± 10.780655, best_reward: 8.486800 ± 15.958512 in #2
Epoch #72: 100001it [05:53, 283.22it/s, env_step=7200000, len=1, loss=1.035, n/ep=0, n/st=10, rew=0.00]                                                                                                                                                   
Epoch #72: test_reward: 4.110946 ± 8.711260, best_reward: 8.486800 ± 15.958512 in #2
Epoch #73: 100001it [05:52, 283.89it/s, env_step=7300000, len=42, loss=1.053, n/ep=0, n/st=10, rew=0.07]                                                                                                                                                  
Epoch #73: test_reward: 4.734265 ± 9.584901, best_reward: 8.486800 ± 15.958512 in #2
Epoch #74: 100001it [05:49, 285.96it/s, env_step=7400000, len=1, loss=1.060, n/ep=0, n/st=10, rew=0.00]                                                                                                                                                   
Epoch #74: test_reward: 6.008546 ± 12.292936, best_reward: 8.486800 ± 15.958512 in #2
Epoch #75: 100001it [05:54, 282.08it/s, env_step=7500000, len=1, loss=1.013, n/ep=0, n/st=10, rew=0.00]                                                                                                                                                   
Epoch #75: test_reward: 3.509600 ± 7.992285, best_reward: 8.486800 ± 15.958512 in #2
Epoch #76: 100001it [05:54, 281.98it/s, env_step=7600000, len=25, loss=1.041, n/ep=0, n/st=10, rew=0.01]                                                                                                                                                  
Epoch #76: test_reward: 4.973378 ± 11.441321, best_reward: 8.486800 ± 15.958512 in #2
Epoch #77: 100001it [05:50, 285.55it/s, env_step=7700000, len=40, loss=1.053, n/ep=1, n/st=10, rew=3.84]                                                                                                                                                  
Epoch #77: test_reward: 3.394243 ± 8.619808, best_reward: 8.486800 ± 15.958512 in #2
Epoch #78: 100001it [05:49, 286.29it/s, env_step=7800000, len=31, loss=1.023, n/ep=0, n/st=10, rew=0.01]                                                                                                                                                  
Epoch #78: test_reward: 4.079859 ± 9.271633, best_reward: 8.486800 ± 15.958512 in #2
Epoch #79: 100001it [05:50, 285.12it/s, env_step=7900000, len=26, loss=1.040, n/ep=0, n/st=10, rew=0.01]                                                                                                                                                  
Epoch #79: test_reward: 4.626345 ± 9.867188, best_reward: 8.486800 ± 15.958512 in #2
Epoch #80: 100001it [05:53, 282.75it/s, env_step=8000000, len=46, loss=1.016, n/ep=0, n/st=10, rew=0.02]                                                                                                                                                  
Epoch #80: test_reward: 3.654252 ± 9.104822, best_reward: 8.486800 ± 15.958512 in #2
{'best_result': '8.49 ± 15.96',
 'best_reward': 8.48679987179565,
 'duration': '28527.82s',
 'test_episode': 8100,
 'test_speed': '587.42 step/s',
 'test_step': 385153,
 'test_time': '655.67s',
 'train_episode': 313523,
 'train_speed': '287.02 step/s',
 'train_step': 8000000,
 'train_time/collector': '13266.69s',
 'train_time/model': '14605.46s'}

# RAINBOW COM NOVOS PESOS

Epoch #1: 100001it [05:58, 279.14it/s, env_step=100000, len=34, loss=1.528, n/ep=0, n/st=10, rew=-0.11]                                                                                                                                                   
Epoch #1: test_reward: 47.022860 ± 82.785887, best_reward: 47.022860 ± 82.785887 in #1
Epoch #2: 100001it [05:56, 280.17it/s, env_step=200000, len=62, loss=1.106, n/ep=1, n/st=10, rew=-0.11]                                                                                                                                                   
Epoch #2: test_reward: 16.165540 ± 51.836385, best_reward: 47.022860 ± 82.785887 in #1
Epoch #3: 100001it [05:58, 279.17it/s, env_step=300000, len=39, loss=0.916, n/ep=0, n/st=10, rew=-0.00]                                                                                                                                                   
Epoch #3: test_reward: 31.858850 ± 67.433608, best_reward: 47.022860 ± 82.785887 in #1
Epoch #4: 100001it [05:54, 281.74it/s, env_step=400000, len=53, loss=0.830, n/ep=0, n/st=10, rew=9.97]                                                                                                                                                    
Epoch #4: test_reward: 35.244410 ± 67.495420, best_reward: 47.022860 ± 82.785887 in #1
Epoch #5: 100001it [05:55, 281.05it/s, env_step=500000, len=34, loss=0.755, n/ep=0, n/st=10, rew=-0.00]                                                                                                                                                   
Epoch #5: test_reward: 36.256970 ± 70.173438, best_reward: 47.022860 ± 82.785887 in #1
Epoch #6: 100001it [05:53, 282.70it/s, env_step=600000, len=1, loss=0.750, n/ep=0, n/st=10, rew=0.00]                                                                                                                                                     
Epoch #6: test_reward: 19.186490 ± 51.012353, best_reward: 47.022860 ± 82.785887 in #1
Epoch #7: 100001it [05:55, 281.56it/s, env_step=700000, len=118, loss=0.746, n/ep=0, n/st=10, rew=94.28]                                                                                                                                                  
Epoch #7: test_reward: 33.315850 ± 67.911154, best_reward: 47.022860 ± 82.785887 in #1
Epoch #8: 100001it [05:57, 279.60it/s, env_step=800000, len=30, loss=0.776, n/ep=1, n/st=10, rew=-0.00]                                                                                                                                                   
Epoch #8: test_reward: 25.446780 ± 56.988481, best_reward: 47.022860 ± 82.785887 in #1
Epoch #9: 100001it [05:54, 282.21it/s, env_step=900000, len=32, loss=0.779, n/ep=0, n/st=10, rew=-0.10]                                                                                                                                                   
Epoch #9: test_reward: 20.209250 ± 44.427894, best_reward: 47.022860 ± 82.785887 in #1
Epoch #10: 100001it [05:55, 281.51it/s, env_step=1000000, len=54, loss=0.734, n/ep=1, n/st=10, rew=-0.20]                                                                                                                                                 
Epoch #10: test_reward: 35.605070 ± 71.654900, best_reward: 47.022860 ± 82.785887 in #1
Epoch #11: 100001it [05:58, 279.15it/s, env_step=1100000, len=1, loss=0.748, n/ep=0, n/st=10, rew=0.00]                                                                                                                                                   
Epoch #11: test_reward: 31.433050 ± 61.654483, best_reward: 47.022860 ± 82.785887 in #1
Epoch #12: 100001it [05:54, 281.84it/s, env_step=1200000, len=88, loss=0.769, n/ep=0, n/st=10, rew=-0.11]                                                                                                                                                 
Epoch #12: test_reward: 21.735460 ± 54.904366, best_reward: 47.022860 ± 82.785887 in #1
Epoch #13: 100001it [05:54, 282.10it/s, env_step=1300000, len=1, loss=0.683, n/ep=1, n/st=10, rew=0.00]                                                                                                                                                   
Epoch #13: test_reward: 17.170550 ± 40.334942, best_reward: 47.022860 ± 82.785887 in #1
Epoch #14: 100001it [05:56, 280.55it/s, env_step=1400000, len=35, loss=0.723, n/ep=0, n/st=10, rew=0.00]                                                                                                                                                  
Epoch #14: test_reward: 23.587600 ± 56.420907, best_reward: 47.022860 ± 82.785887 in #1
Epoch #15: 100001it [05:53, 283.11it/s, env_step=1500000, len=44, loss=0.727, n/ep=2, n/st=10, rew=27.47]                                                                                                                                                 
Epoch #15: test_reward: 26.138310 ± 59.362236, best_reward: 47.022860 ± 82.785887 in #1
Epoch #16: 100001it [05:53, 282.93it/s, env_step=1600000, len=42, loss=0.720, n/ep=0, n/st=10, rew=0.00]                                                                                                                                                  
Epoch #16: test_reward: 38.702350 ± 68.340987, best_reward: 47.022860 ± 82.785887 in #1
Epoch #17: 100001it [05:52, 283.77it/s, env_step=1700000, len=5, loss=0.768, n/ep=0, n/st=10, rew=2.60]                                                                                                                                                   
Epoch #17: test_reward: 39.945700 ± 77.091218, best_reward: 47.022860 ± 82.785887 in #1
Epoch #18: 100001it [05:55, 281.00it/s, env_step=1800000, len=1, loss=0.706, n/ep=0, n/st=10, rew=0.00]                                                                                                                                                   
Epoch #18: test_reward: 24.006520 ± 55.376979, best_reward: 47.022860 ± 82.785887 in #1
Epoch #19: 100001it [05:55, 280.94it/s, env_step=1900000, len=17, loss=0.702, n/ep=0, n/st=10, rew=4.90]                                                                                                                                                  
Epoch #19: test_reward: 20.433850 ± 44.171403, best_reward: 47.022860 ± 82.785887 in #1
Epoch #20: 100001it [05:54, 282.37it/s, env_step=2000000, len=1, loss=0.745, n/ep=0, n/st=10, rew=0.00]                                                                                                                                                   
Epoch #20: test_reward: 23.559930 ± 57.552863, best_reward: 47.022860 ± 82.785887 in #1
Epoch #21: 100001it [05:56, 280.71it/s, env_step=2100000, len=35, loss=0.727, n/ep=0, n/st=10, rew=-0.00]                                                                                                                                                 
Epoch #21: test_reward: 27.830070 ± 67.303802, best_reward: 47.022860 ± 82.785887 in #1
Epoch #22: 100001it [05:52, 284.05it/s, env_step=2200000, len=58, loss=0.663, n/ep=1, n/st=10, rew=-0.00]                                                                                                                                                 
Epoch #22: test_reward: 19.044190 ± 35.771846, best_reward: 47.022860 ± 82.785887 in #1
Epoch #23: 100001it [05:51, 284.13it/s, env_step=2300000, len=1, loss=0.694, n/ep=0, n/st=10, rew=0.00]                                                                                                                                                   
Epoch #23: test_reward: 24.030790 ± 61.295761, best_reward: 47.022860 ± 82.785887 in #1
Epoch #24: 100001it [05:54, 282.09it/s, env_step=2400000, len=40, loss=0.704, n/ep=0, n/st=10, rew=0.00]                                                                                                                                                  
Epoch #24: test_reward: 24.953910 ± 51.800992, best_reward: 47.022860 ± 82.785887 in #1
Epoch #25: 100001it [05:55, 281.21it/s, env_step=2500000, len=36, loss=0.688, n/ep=0, n/st=10, rew=-0.10]                                                                                                                                                 
Epoch #25: test_reward: 33.223380 ± 63.125918, best_reward: 47.022860 ± 82.785887 in #1
Epoch #26: 100001it [05:53, 283.20it/s, env_step=2600000, len=85, loss=0.690, n/ep=1, n/st=10, rew=-0.10]                                                                                                                                                 
Epoch #26: test_reward: 19.881670 ± 53.087648, best_reward: 47.022860 ± 82.785887 in #1
Epoch #27: 100001it [05:56, 280.24it/s, env_step=2700000, len=34, loss=0.710, n/ep=0, n/st=10, rew=0.00]                                                                                                                                                  
Epoch #27: test_reward: 15.426330 ± 37.820052, best_reward: 47.022860 ± 82.785887 in #1
Epoch #28: 100001it [05:54, 281.93it/s, env_step=2800000, len=28, loss=0.707, n/ep=0, n/st=10, rew=-0.05]                                                                                                                                                 
Epoch #28: test_reward: 28.328260 ± 63.967973, best_reward: 47.022860 ± 82.785887 in #1
Epoch #29: 100001it [05:57, 279.54it/s, env_step=2900000, len=59, loss=0.708, n/ep=0, n/st=10, rew=-0.11]                                                                                                                                                 
Epoch #29: test_reward: 27.171150 ± 60.507016, best_reward: 47.022860 ± 82.785887 in #1
Epoch #30: 100001it [05:54, 282.01it/s, env_step=3000000, len=28, loss=0.686, n/ep=0, n/st=10, rew=0.10]                                                                                                                                                  
Epoch #30: test_reward: 22.590460 ± 57.128836, best_reward: 47.022860 ± 82.785887 in #1
Epoch #31: 100001it [05:56, 280.45it/s, env_step=3100000, len=50, loss=0.653, n/ep=4, n/st=10, rew=42.34]                                                                                                                                                 
Epoch #31: test_reward: 36.715870 ± 66.166588, best_reward: 47.022860 ± 82.785887 in #1
Epoch #32: 100001it [05:55, 281.10it/s, env_step=3200000, len=1, loss=0.713, n/ep=0, n/st=10, rew=0.00]                                                                                                                                                   
Epoch #32: test_reward: 25.819990 ± 60.139632, best_reward: 47.022860 ± 82.785887 in #1
Epoch #33: 100001it [05:52, 283.49it/s, env_step=3300000, len=1, loss=0.680, n/ep=0, n/st=10, rew=0.00]                                                                                                                                                   
Epoch #33: test_reward: 31.256470 ± 71.350065, best_reward: 47.022860 ± 82.785887 in #1
Epoch #34: 100001it [05:52, 283.57it/s, env_step=3400000, len=57, loss=0.683, n/ep=0, n/st=10, rew=-0.21]                                                                                                                                                 
Epoch #34: test_reward: 29.815280 ± 63.201790, best_reward: 47.022860 ± 82.785887 in #1
Epoch #35: 100001it [05:53, 282.99it/s, env_step=3500000, len=1, loss=0.676, n/ep=5, n/st=10, rew=0.00]                                                                                                                                                   
Epoch #35: test_reward: 32.250250 ± 64.995391, best_reward: 47.022860 ± 82.785887 in #1
Epoch #36: 100001it [05:53, 283.26it/s, env_step=3600000, len=16, loss=0.669, n/ep=2, n/st=10, rew=-0.05]                                                                                                                                                 
Epoch #36: test_reward: 25.373150 ± 61.805447, best_reward: 47.022860 ± 82.785887 in #1
Epoch #37: 100001it [05:53, 282.94it/s, env_step=3700000, len=42, loss=0.700, n/ep=0, n/st=10, rew=36.64]                                                                                                                                                 
Epoch #37: test_reward: 25.540420 ± 51.878293, best_reward: 47.022860 ± 82.785887 in #1
Epoch #38: 100001it [05:56, 280.22it/s, env_step=3800000, len=42, loss=0.694, n/ep=2, n/st=10, rew=-0.06]                                                                                                                                                 
Epoch #38: test_reward: 17.828600 ± 45.960407, best_reward: 47.022860 ± 82.785887 in #1
Epoch #39: 100001it [05:54, 281.85it/s, env_step=3900000, len=1, loss=0.707, n/ep=0, n/st=10, rew=0.00]                                                                                                                                                   
Epoch #39: test_reward: 34.199330 ± 70.525841, best_reward: 47.022860 ± 82.785887 in #1
Epoch #40: 100001it [05:55, 281.48it/s, env_step=4000000, len=13, loss=0.684, n/ep=0, n/st=10, rew=-0.00]                                                                                                                                                 
Epoch #40: test_reward: 29.408690 ± 62.740851, best_reward: 47.022860 ± 82.785887 in #1
Epoch #41: 100001it [05:55, 280.95it/s, env_step=4100000, len=28, loss=0.705, n/ep=0, n/st=10, rew=0.00]                                                                                                                                                  
Epoch #41: test_reward: 27.117860 ± 60.024539, best_reward: 47.022860 ± 82.785887 in #1
Epoch #42: 100001it [05:53, 283.05it/s, env_step=4200000, len=25, loss=0.689, n/ep=0, n/st=10, rew=-0.00]                                                                                                                                                 
Epoch #42: test_reward: 30.804080 ± 62.179215, best_reward: 47.022860 ± 82.785887 in #1
Epoch #43: 100001it [05:57, 279.70it/s, env_step=4300000, len=29, loss=0.678, n/ep=0, n/st=10, rew=-0.00]                                                                                                                                                 
Epoch #43: test_reward: 32.890260 ± 60.025757, best_reward: 47.022860 ± 82.785887 in #1
Epoch #44: 100001it [05:56, 280.35it/s, env_step=4400000, len=1, loss=0.735, n/ep=5, n/st=10, rew=0.00]                                                                                                                                                   
Epoch #44: test_reward: 25.073230 ± 52.075308, best_reward: 47.022860 ± 82.785887 in #1
Epoch #45: 100001it [05:53, 282.53it/s, env_step=4500000, len=37, loss=0.679, n/ep=1, n/st=10, rew=-0.20]                                                                                                                                                 
Epoch #45: test_reward: 33.687240 ± 59.157324, best_reward: 47.022860 ± 82.785887 in #1
Epoch #46: 100001it [05:59, 278.31it/s, env_step=4600000, len=32, loss=0.662, n/ep=0, n/st=10, rew=-0.10]                                                                                                                                                 
Epoch #46: test_reward: 25.718560 ± 58.677368, best_reward: 47.022860 ± 82.785887 in #1
Epoch #47: 100001it [05:56, 280.78it/s, env_step=4700000, len=10, loss=0.698, n/ep=4, n/st=10, rew=-0.00]                                                                                                                                                 
Epoch #47: test_reward: 16.543830 ± 42.425924, best_reward: 47.022860 ± 82.785887 in #1
Epoch #48: 100001it [05:56, 280.58it/s, env_step=4800000, len=71, loss=0.685, n/ep=0, n/st=10, rew=34.69]                                                                                                                                                 
Epoch #48: test_reward: 24.741650 ± 57.243962, best_reward: 47.022860 ± 82.785887 in #1
Epoch #49: 100001it [05:56, 280.80it/s, env_step=4900000, len=45, loss=0.693, n/ep=0, n/st=10, rew=14.92]                                                                                                                                                 
Epoch #49: test_reward: 30.064550 ± 59.924818, best_reward: 47.022860 ± 82.785887 in #1
Epoch #50: 100001it [05:57, 280.07it/s, env_step=5000000, len=1, loss=0.705, n/ep=0, n/st=10, rew=0.00]                                                                                                                                                   
Epoch #50: test_reward: 25.314700 ± 53.562597, best_reward: 47.022860 ± 82.785887 in #1
Epoch #51: 100001it [05:56, 280.63it/s, env_step=5100000, len=34, loss=0.718, n/ep=0, n/st=10, rew=-0.00]                                                                                                                                                 
Epoch #51: test_reward: 20.120800 ± 48.581030, best_reward: 47.022860 ± 82.785887 in #1
Epoch #52: 100001it [05:54, 281.92it/s, env_step=5200000, len=15, loss=0.677, n/ep=0, n/st=10, rew=-0.00]                                                                                                                                                 
Epoch #52: test_reward: 19.980840 ± 54.983660, best_reward: 47.022860 ± 82.785887 in #1
Epoch #53: 100001it [05:54, 282.10it/s, env_step=5300000, len=40, loss=0.690, n/ep=0, n/st=10, rew=-0.21]                                                                                                                                                 
Epoch #53: test_reward: 25.318290 ± 58.046583, best_reward: 47.022860 ± 82.785887 in #1
Epoch #54: 100001it [05:58, 279.22it/s, env_step=5400000, len=69, loss=0.715, n/ep=0, n/st=10, rew=-0.01]                                                                                                                                                 
Epoch #54: test_reward: 22.532520 ± 60.099581, best_reward: 47.022860 ± 82.785887 in #1
Epoch #55: 100001it [05:54, 282.34it/s, env_step=5500000, len=38, loss=0.692, n/ep=0, n/st=10, rew=14.70]                                                                                                                                                 
Epoch #55: test_reward: 27.178350 ± 60.859977, best_reward: 47.022860 ± 82.785887 in #1
Epoch #56: 100001it [05:54, 281.77it/s, env_step=5600000, len=15, loss=0.702, n/ep=0, n/st=10, rew=-0.00]                                                                                                                                                 
Epoch #56: test_reward: 25.767340 ± 57.389632, best_reward: 47.022860 ± 82.785887 in #1
Epoch #57: 100001it [05:56, 280.32it/s, env_step=5700000, len=25, loss=0.706, n/ep=4, n/st=10, rew=22.46]                                                                                                                                                 
Epoch #57: test_reward: 28.100470 ± 58.044854, best_reward: 47.022860 ± 82.785887 in #1
Epoch #58: 100001it [05:55, 281.30it/s, env_step=5800000, len=35, loss=0.709, n/ep=0, n/st=10, rew=-0.10]                                                                                                                                                 
Epoch #58: test_reward: 19.317050 ± 45.798993, best_reward: 47.022860 ± 82.785887 in #1
Epoch #59: 100001it [05:54, 282.40it/s, env_step=5900000, len=31, loss=0.714, n/ep=0, n/st=10, rew=-0.00]                                                                                                                                                 
Epoch #59: test_reward: 21.546410 ± 43.537711, best_reward: 47.022860 ± 82.785887 in #1
Epoch #60: 100001it [05:59, 277.98it/s, env_step=6000000, len=32, loss=0.684, n/ep=1, n/st=10, rew=-0.10]                                                                                                                                                 
Epoch #60: test_reward: 23.335120 ± 54.699462, best_reward: 47.022860 ± 82.785887 in #1
Epoch #61: 100001it [05:56, 280.55it/s, env_step=6100000, len=1, loss=0.710, n/ep=2, n/st=10, rew=0.00]                                                                                                                                                   
Epoch #61: test_reward: 14.000360 ± 37.864177, best_reward: 47.022860 ± 82.785887 in #1
Epoch #62: 100001it [05:59, 278.41it/s, env_step=6200000, len=1, loss=0.700, n/ep=0, n/st=10, rew=0.00]                                                                                                                                                   
Epoch #62: test_reward: 29.129520 ± 60.488496, best_reward: 47.022860 ± 82.785887 in #1
Epoch #63: 100001it [05:56, 280.77it/s, env_step=6300000, len=31, loss=0.685, n/ep=0, n/st=10, rew=20.00]                                                                                                                                                 
Epoch #63: test_reward: 12.847480 ± 37.305834, best_reward: 47.022860 ± 82.785887 in #1
Epoch #64: 100001it [05:53, 282.56it/s, env_step=6400000, len=52, loss=0.690, n/ep=0, n/st=10, rew=-0.11]                                                                                                                                                 
Epoch #64: test_reward: 37.039080 ± 72.817810, best_reward: 47.022860 ± 82.785887 in #1
Epoch #65: 100001it [05:53, 283.03it/s, env_step=6500000, len=36, loss=0.692, n/ep=0, n/st=10, rew=21.65]                                                                                                                                                 
Epoch #65: test_reward: 33.588330 ± 61.977307, best_reward: 47.022860 ± 82.785887 in #1
Epoch #66: 100001it [05:55, 281.34it/s, env_step=6600000, len=1, loss=0.695, n/ep=0, n/st=10, rew=0.00]                                                                                                                                                   
Epoch #66: test_reward: 34.141950 ± 64.415813, best_reward: 47.022860 ± 82.785887 in #1
Epoch #67: 100001it [05:58, 279.33it/s, env_step=6700000, len=32, loss=0.676, n/ep=0, n/st=10, rew=-0.10]                                                                                                                                                 
Epoch #67: test_reward: 28.807350 ± 58.734634, best_reward: 47.022860 ± 82.785887 in #1
Epoch #68: 100001it [05:55, 281.08it/s, env_step=6800000, len=1, loss=0.673, n/ep=0, n/st=10, rew=0.00]                                                                                                                                                   
Epoch #68: test_reward: 37.861980 ± 79.762233, best_reward: 47.022860 ± 82.785887 in #1
Epoch #69: 100001it [05:53, 283.10it/s, env_step=6900000, len=75, loss=0.700, n/ep=0, n/st=10, rew=-0.21]                                                                                                                                                 
Epoch #69: test_reward: 27.342370 ± 62.243125, best_reward: 47.022860 ± 82.785887 in #1
Epoch #70: 100001it [05:53, 282.72it/s, env_step=7000000, len=1, loss=0.691, n/ep=0, n/st=10, rew=0.00]                                                                                                                                                   
Epoch #70: test_reward: 19.041070 ± 39.956102, best_reward: 47.022860 ± 82.785887 in #1
Epoch #71: 100001it [05:56, 280.60it/s, env_step=7100000, len=1, loss=0.705, n/ep=0, n/st=10, rew=0.00]                                                                                                                                                   
Epoch #71: test_reward: 23.226710 ± 58.561923, best_reward: 47.022860 ± 82.785887 in #1
Epoch #72: 100001it [05:56, 280.89it/s, env_step=7200000, len=34, loss=0.664, n/ep=0, n/st=10, rew=24.79]                                                                                                                                                 
Epoch #72: test_reward: 33.262170 ± 68.767386, best_reward: 47.022860 ± 82.785887 in #1
Epoch #73: 100001it [05:54, 282.15it/s, env_step=7300000, len=18, loss=0.690, n/ep=0, n/st=10, rew=15.05]                                                                                                                                                 
Epoch #73: test_reward: 33.548070 ± 69.036974, best_reward: 47.022860 ± 82.785887 in #1
Epoch #74: 100001it [05:55, 281.57it/s, env_step=7400000, len=37, loss=0.712, n/ep=0, n/st=10, rew=-0.11]                                                                                                                                                 
Epoch #74: test_reward: 26.572370 ± 59.350526, best_reward: 47.022860 ± 82.785887 in #1
Epoch #75: 100001it [05:55, 281.10it/s, env_step=7500000, len=52, loss=0.705, n/ep=0, n/st=10, rew=-0.21]                                                                                                                                                 
Epoch #75: test_reward: 29.280610 ± 65.258213, best_reward: 47.022860 ± 82.785887 in #1
Epoch #76: 100001it [05:52, 283.96it/s, env_step=7600000, len=69, loss=0.714, n/ep=0, n/st=10, rew=0.05]                                                                                                                                                  
Epoch #76: test_reward: 27.330860 ± 66.901542, best_reward: 47.022860 ± 82.785887 in #1
Epoch #77: 100001it [05:49, 286.44it/s, env_step=7700000, len=25, loss=0.731, n/ep=0, n/st=10, rew=10.20]                                                                                                                                                 
Epoch #77: test_reward: 22.816010 ± 48.269999, best_reward: 47.022860 ± 82.785887 in #1
Epoch #78: 100001it [05:51, 284.75it/s, env_step=7800000, len=27, loss=0.742, n/ep=1, n/st=10, rew=-0.00]                                                                                                                                                 
Epoch #78: test_reward: 25.886760 ± 55.178169, best_reward: 47.022860 ± 82.785887 in #1
Epoch #79: 100001it [05:56, 280.29it/s, env_step=7900000, len=1, loss=0.692, n/ep=0, n/st=10, rew=0.00]                                                                                                                                                   
Epoch #79: test_reward: 20.107760 ± 53.061941, best_reward: 47.022860 ± 82.785887 in #1
Epoch #80: 100001it [05:51, 284.49it/s, env_step=8000000, len=10, loss=0.711, n/ep=4, n/st=10, rew=-0.03]                                                                                                                                                 
Epoch #80: test_reward: 25.336480 ± 58.646951, best_reward: 47.022860 ± 82.785887 in #1
{'best_result': '47.02 ± 82.79',
 'best_reward': 47.02285999999998,
 'duration': '29450.69s',
 'test_episode': 8100,
 'test_speed': '579.32 step/s',
 'test_step': 598038,
 'test_time': '1032.31s',
 'train_episode': 321707,
 'train_speed': '281.51 step/s',
 'train_step': 8000000,
 'train_time/collector': '13663.99s',
 'train_time/model': '14754.39s'}

# DTQN

uv run levd.py --algorithm dtqn --scenario full_deathmatch --train_levels 0 --train_maps 1 2 --test_levels 0 --test_maps 4 11 --seed 50 --epoch 50 --step-per-collect 10 --device cuda --batch-size 64

Run: 50_20251217_135424

Epoch #1: 100001it [05:37, 296.68it/s, env_step=100000, len=57, loss=0.124, n/ep=0, n/st=10, rew=4.06]                                                                             
Epoch #1: test_reward: 14.891028 ± 47.278584, best_reward: 14.891028 ± 47.278584 in #1
Epoch #2: 100001it [05:38, 295.19it/s, env_step=200000, len=44, loss=0.141, n/ep=0, n/st=10, rew=2.17]                                                                             
Epoch #2: test_reward: 15.198612 ± 39.677467, best_reward: 15.198612 ± 39.677467 in #2
Epoch #3: 100001it [05:38, 295.50it/s, env_step=300000, len=30, loss=0.160, n/ep=0, n/st=10, rew=1.78]                                                                             
Epoch #3: test_reward: 9.000279 ± 28.341648, best_reward: 15.198612 ± 39.677467 in #2
Epoch #4: 100001it [05:40, 294.05it/s, env_step=400000, len=40, loss=0.177, n/ep=0, n/st=10, rew=3.24]                                                                             
Epoch #4: test_reward: 11.422879 ± 39.096205, best_reward: 15.198612 ± 39.677467 in #2
Epoch #5: 100001it [05:39, 294.13it/s, env_step=500000, len=19, loss=0.259, n/ep=3, n/st=10, rew=0.49]                                                                             
Epoch #5: test_reward: 24.936581 ± 60.144270, best_reward: 24.936581 ± 60.144270 in #5
Epoch #6: 100001it [05:39, 294.42it/s, env_step=600000, len=25, loss=0.288, n/ep=0, n/st=10, rew=1.36]                                                                             
Epoch #6: test_reward: 12.770102 ± 41.819768, best_reward: 24.936581 ± 60.144270 in #5
Epoch #7: 100001it [05:39, 294.77it/s, env_step=700000, len=94, loss=0.341, n/ep=0, n/st=10, rew=4.95]                                                                             
Epoch #7: test_reward: 22.668581 ± 49.412517, best_reward: 24.936581 ± 60.144270 in #5
Epoch #8: 100001it [05:38, 295.78it/s, env_step=800000, len=1, loss=0.436, n/ep=0, n/st=10, rew=0.00]                                                                              
Epoch #8: test_reward: 12.819776 ± 43.510464, best_reward: 24.936581 ± 60.144270 in #5
Epoch #9: 100001it [05:38, 295.80it/s, env_step=900000, len=118, loss=0.552, n/ep=0, n/st=10, rew=9.38]                                                                            
Epoch #9: test_reward: 8.214018 ± 30.872696, best_reward: 24.936581 ± 60.144270 in #5
Epoch #10: 100001it [05:35, 298.01it/s, env_step=1000000, len=156, loss=0.574, n/ep=0, n/st=10, rew=15.36]                                                                         
Epoch #10: test_reward: 17.044621 ± 49.608463, best_reward: 24.936581 ± 60.144270 in #5
Epoch #11: 100001it [05:35, 298.43it/s, env_step=1100000, len=364, loss=0.753, n/ep=0, n/st=10, rew=31.25]                                                                         
Epoch #11: test_reward: 15.499849 ± 44.473363, best_reward: 24.936581 ± 60.144270 in #5
Epoch #12: 100001it [05:35, 298.35it/s, env_step=1200000, len=1, loss=0.793, n/ep=0, n/st=10, rew=0.00]                                                                            
Epoch #12: test_reward: 16.537075 ± 83.458564, best_reward: 24.936581 ± 60.144270 in #5
Epoch #13: 100001it [05:34, 298.96it/s, env_step=1300000, len=60, loss=0.926, n/ep=1, n/st=10, rew=6.16]                                                                           
Epoch #13: test_reward: 13.079738 ± 46.547772, best_reward: 24.936581 ± 60.144270 in #5
Epoch #14: 100001it [05:34, 298.88it/s, env_step=1400000, len=1, loss=0.943, n/ep=0, n/st=10, rew=0.00]                                                                            
Epoch #14: test_reward: 6.878771 ± 31.078641, best_reward: 24.936581 ± 60.144270 in #5
Epoch #15: 100001it [05:34, 299.05it/s, env_step=1500000, len=82, loss=1.188, n/ep=0, n/st=10, rew=7.97]                                                                           
Epoch #15: test_reward: 14.737506 ± 68.761111, best_reward: 24.936581 ± 60.144270 in #5
Epoch #16: 100001it [05:35, 298.24it/s, env_step=1600000, len=54, loss=1.076, n/ep=0, n/st=10, rew=5.66]                                                                           
Epoch #16: test_reward: 9.326986 ± 33.929916, best_reward: 24.936581 ± 60.144270 in #5
Epoch #17: 100001it [05:34, 299.12it/s, env_step=1700000, len=72, loss=1.236, n/ep=0, n/st=10, rew=6.32]                                                                           
Epoch #17: test_reward: 6.245741 ± 21.627330, best_reward: 24.936581 ± 60.144270 in #5
Epoch #18: 100001it [05:35, 298.09it/s, env_step=1800000, len=120, loss=1.155, n/ep=0, n/st=10, rew=10.66]                                                                         
Epoch #18: test_reward: 11.560268 ± 36.716488, best_reward: 24.936581 ± 60.144270 in #5
Epoch #19: 100001it [05:35, 298.13it/s, env_step=1900000, len=140, loss=1.317, n/ep=0, n/st=10, rew=10.25]                                                                         
Epoch #19: test_reward: 4.789904 ± 21.636273, best_reward: 24.936581 ± 60.144270 in #5
Epoch #20: 100001it [05:33, 299.60it/s, env_step=2000000, len=29, loss=1.121, n/ep=0, n/st=10, rew=2.94]                                                                           
Epoch #20: test_reward: 8.502202 ± 34.330683, best_reward: 24.936581 ± 60.144270 in #5
Epoch #21: 100001it [05:35, 298.18it/s, env_step=2100000, len=270, loss=1.234, n/ep=0, n/st=10, rew=25.60]                                                                         
Epoch #21: test_reward: 8.733165 ± 29.899928, best_reward: 24.936581 ± 60.144270 in #5
Epoch #22: 100001it [05:32, 301.00it/s, env_step=2200000, len=140, loss=1.250, n/ep=0, n/st=10, rew=11.60]                                                                         
Epoch #22: test_reward: 20.461977 ± 56.106932, best_reward: 24.936581 ± 60.144270 in #5
Epoch #23: 100001it [05:34, 299.05it/s, env_step=2300000, len=103, loss=1.367, n/ep=0, n/st=10, rew=9.24]                                                                          
Epoch #23: test_reward: 3.405906 ± 16.958300, best_reward: 24.936581 ± 60.144270 in #5
Epoch #24: 100001it [05:34, 298.84it/s, env_step=2400000, len=1, loss=1.125, n/ep=0, n/st=10, rew=0.00]                                                                            
Epoch #24: test_reward: 12.198504 ± 38.818733, best_reward: 24.936581 ± 60.144270 in #5
Epoch #25: 100001it [05:35, 298.25it/s, env_step=2500000, len=178, loss=1.248, n/ep=0, n/st=10, rew=16.89]                                                                         
Epoch #25: test_reward: 7.375741 ± 28.338831, best_reward: 24.936581 ± 60.144270 in #5
Epoch #26: 100001it [05:34, 299.11it/s, env_step=2600000, len=13, loss=1.220, n/ep=0, n/st=10, rew=0.95]                                                                           
Epoch #26: test_reward: 10.359052 ± 36.516033, best_reward: 24.936581 ± 60.144270 in #5
Epoch #27: 100001it [05:34, 299.10it/s, env_step=2700000, len=1, loss=1.159, n/ep=0, n/st=10, rew=0.00]                                                                            
Epoch #27: test_reward: 8.920011 ± 31.909933, best_reward: 24.936581 ± 60.144270 in #5
Epoch #28: 100001it [05:35, 298.33it/s, env_step=2800000, len=14, loss=1.162, n/ep=0, n/st=10, rew=1.14]                                                                           
Epoch #28: test_reward: 13.050926 ± 50.149019, best_reward: 24.936581 ± 60.144270 in #5
Epoch #29: 100001it [05:33, 299.65it/s, env_step=2900000, len=16, loss=1.287, n/ep=0, n/st=10, rew=0.76]                                                                           
Epoch #29: test_reward: 13.315451 ± 39.101818, best_reward: 24.936581 ± 60.144270 in #5
Epoch #30: 100001it [05:34, 299.22it/s, env_step=3000000, len=28, loss=1.281, n/ep=0, n/st=10, rew=2.24]                                                                           
Epoch #30: test_reward: 10.398501 ± 37.876372, best_reward: 24.936581 ± 60.144270 in #5
Epoch #31: 100001it [05:34, 299.08it/s, env_step=3100000, len=29, loss=1.314, n/ep=0, n/st=10, rew=1.96]                                                                           
Epoch #31: test_reward: 9.854548 ± 28.623096, best_reward: 24.936581 ± 60.144270 in #5
Epoch #32: 100001it [05:35, 298.40it/s, env_step=3200000, len=45, loss=1.348, n/ep=0, n/st=10, rew=2.70]                                                                           
Epoch #32: test_reward: 16.054603 ± 39.563296, best_reward: 24.936581 ± 60.144270 in #5
Epoch #33: 100001it [05:34, 298.82it/s, env_step=3300000, len=88, loss=1.369, n/ep=0, n/st=10, rew=5.89]                                                                           
Epoch #33: test_reward: 7.934618 ± 27.525506, best_reward: 24.936581 ± 60.144270 in #5
Epoch #34: 100001it [05:39, 294.90it/s, env_step=3400000, len=25, loss=1.260, n/ep=0, n/st=10, rew=2.18]                                                                           
Epoch #34: test_reward: 10.357871 ± 35.729769, best_reward: 24.936581 ± 60.144270 in #5
Epoch #35: 100001it [05:38, 295.22it/s, env_step=3500000, len=7, loss=1.506, n/ep=0, n/st=10, rew=0.62]                                                                            
Epoch #35: test_reward: 9.595692 ± 32.421627, best_reward: 24.936581 ± 60.144270 in #5
Epoch #36: 100001it [05:34, 299.08it/s, env_step=3600000, len=63, loss=1.400, n/ep=0, n/st=10, rew=3.70]                                                                           
Epoch #36: test_reward: 13.313469 ± 43.538895, best_reward: 24.936581 ± 60.144270 in #5
Epoch #37: 100001it [05:35, 297.73it/s, env_step=3700000, len=42, loss=1.354, n/ep=1, n/st=10, rew=2.90]                                                                           
Epoch #37: test_reward: 22.743571 ± 51.916712, best_reward: 24.936581 ± 60.144270 in #5
Epoch #38: 100001it [05:32, 300.89it/s, env_step=3800000, len=1, loss=1.203, n/ep=0, n/st=10, rew=0.00]                                                                            
Epoch #38: test_reward: 15.032532 ± 41.874876, best_reward: 24.936581 ± 60.144270 in #5
Epoch #39: 100001it [05:30, 303.02it/s, env_step=3900000, len=63, loss=1.333, n/ep=0, n/st=10, rew=4.66]                                                                           
Epoch #39: test_reward: 16.825819 ± 45.318670, best_reward: 24.936581 ± 60.144270 in #5
Epoch #40: 100001it [05:33, 300.09it/s, env_step=4000000, len=71, loss=1.438, n/ep=0, n/st=10, rew=7.22]                                                                           
Epoch #40: test_reward: 22.572648 ± 78.564336, best_reward: 24.936581 ± 60.144270 in #5
Epoch #41: 100001it [05:30, 302.21it/s, env_step=4100000, len=52, loss=1.465, n/ep=0, n/st=10, rew=5.39]                                                                           
Epoch #41: test_reward: 6.025397 ± 40.858840, best_reward: 24.936581 ± 60.144270 in #5
Epoch #42: 100001it [05:31, 301.62it/s, env_step=4200000, len=240, loss=1.527, n/ep=0, n/st=10, rew=18.06]                                                                         
Epoch #42: test_reward: 1.257883 ± 12.515778, best_reward: 24.936581 ± 60.144270 in #5
Epoch #43: 100001it [05:30, 302.70it/s, env_step=4300000, len=29, loss=1.275, n/ep=0, n/st=10, rew=2.49]                                                                           
Epoch #43: test_reward: 26.295062 ± 59.684313, best_reward: 26.295062 ± 59.684313 in #43
Epoch #44: 100001it [05:31, 302.11it/s, env_step=4400000, len=1, loss=1.277, n/ep=0, n/st=10, rew=0.00]                                                                            
Epoch #44: test_reward: 8.503649 ± 29.317777, best_reward: 26.295062 ± 59.684313 in #43
Epoch #45: 100001it [05:30, 302.23it/s, env_step=4500000, len=24, loss=1.303, n/ep=0, n/st=10, rew=2.16]                                                                           
Epoch #45: test_reward: 15.155682 ± 44.796474, best_reward: 26.295062 ± 59.684313 in #43
Epoch #46: 100001it [05:30, 302.58it/s, env_step=4600000, len=85, loss=1.240, n/ep=0, n/st=10, rew=6.23]                                                                           
Epoch #46: test_reward: 16.779617 ± 42.758787, best_reward: 26.295062 ± 59.684313 in #43
Epoch #47: 100001it [05:30, 302.58it/s, env_step=4700000, len=69, loss=1.242, n/ep=0, n/st=10, rew=5.20]                                                                           
Epoch #47: test_reward: 8.196857 ± 29.829652, best_reward: 26.295062 ± 59.684313 in #43
Epoch #48: 100001it [05:31, 301.21it/s, env_step=4800000, len=60, loss=1.270, n/ep=0, n/st=10, rew=3.45]                                                                           
Epoch #48: test_reward: 7.745544 ± 28.579750, best_reward: 26.295062 ± 59.684313 in #43
Epoch #49: 100001it [05:36, 297.43it/s, env_step=4900000, len=36, loss=1.408, n/ep=0, n/st=10, rew=2.35]                                                                           
Epoch #49: test_reward: 13.924335 ± 41.493322, best_reward: 26.295062 ± 59.684313 in #43
Epoch #50: 100001it [05:34, 298.57it/s, env_step=5000000, len=70, loss=1.275, n/ep=1, n/st=10, rew=5.97]                                                                           
Epoch #50: test_reward: 15.423759 ± 44.087248, best_reward: 26.295062 ± 59.684313 in #43
{'best_result': '26.30 ± 59.68',
 'best_reward': 26.295062474387187,
 'duration': '17095.84s',
 'test_episode': 5100,
 'test_speed': '289.46 step/s',
 'test_step': 100934,
 'test_time': '348.70s',
 'train_episode': 70778,
 'train_speed': '298.56 step/s',
 'train_step': 5000000,
 'train_time/collector': '8133.06s',
 'train_time/model': '8614.08s'}
