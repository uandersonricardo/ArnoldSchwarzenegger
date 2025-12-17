--- BASELINE

uv run levd.py --watch --algorithm rainbow --scenario full_deathmatch --resume-path 812_20251209_004624_Initial_levdoom_model/policy_best.pth --test_levels 0 --test_maps 4

=== EVALUATIONS cig ===
[(1, 134, np.float64(32.0), 1, 2, 2), (2, 253, np.float64(53.0), 1, 3, 1), (3, 221, np.float64(43.0), 1, 4, 2), (4, 179, np.float64(42.0), 1, 4, 2), (5, 103, np.float64(24.0), 1, 3, 1), (6, 28, np.float64(4.0), 1, 1, 1), (7, 103, np.float64(20.0), 1, 2, 1), (8, 65, np.float64(15.0), 1, 2, 2), (9, 70, np.float64(19.0), 1, 2, 1), (10, 72, np.float64(23.0), 1, 3, 1), (11, 188, np.float64(42.0), 1, 4, 2), (12, 43, np.float64(15.0), 1, 2, 1), (13, 42, np.float64(11.0), 1, 2, 1), (14, 192, np.float64(42.0), 1, 3, 1), (15, 98, np.float64(27.0), 1, 2, 1), (16, 123, np.float64(30.0), 1, 4, 2), (17, 169, np.float64(35.0), 1, 2, 1), (18, 60, np.float64(21.0), 1, 1, 2), (19, 51, np.float64(17.0), 1, 2, 1), (20, 139, np.float64(29.0), 1, 4, 0), (21, 125, np.float64(37.0), 1, 4, 2), (22, 165, np.float64(36.0), 1, 0, 0), (23, 197, np.float64(47.0), 1, 3, 2), (24, 24, np.float64(2.0), 1, 2, 1), (25, 283, np.float64(57.0), 1, 5, 2), (26, 194, np.float64(40.0), 1, 5, 1), (27, 59, np.float64(11.0), 1, 2, 1), (28, 170, np.float64(41.0), 1, 4, 2), (29, 183, np.float64(37.0), 1, 5, 1), (30, 58, np.float64(14.0), 1, 0, 0)]

=== TABLE ===
Total episodes: 30
Total frames survived: 3791
Total kills: 866.0
Total deaths: 30
K/D Ratio: 28.866666666666667
Total items collected: 82
Total weapons collected: 38

Average frames survived: 126.36666666666666
Average kills: 28.866666666666667
Average deaths: 1.0
Average K/D Ratio: 28.866666666666667
Average items collected: 2.7333333333333334
Average weapons collected: 1.2666666666666666

---

=== EVALUATIONS cig ===
[(1, 64, np.float64(9.0), 1, 4, 1), (2, 85, np.float64(6.0), 1, 6, 0), (3, 96, np.float64(11.0), 1, 1, 1), (4, 72, np.float64(11.0), 1, 3, 1), (5, 108, np.float64(11.0), 1, 4, 0), (6, 118, np.float64(15.0), 1, 7, 1), (7, 109, np.float64(15.0), 1, 2, 1), (8, 130, np.float64(18.0), 1, 4, 1), (9, 124, np.float64(16.0), 1, 2, 0), (10, 67, np.float64(10.0), 1, 1, 1), (11, 104, np.float64(12.0), 1, 3, 0), (12, 181, np.float64(20.0), 1, 2, 0), (13, 265, np.float64(29.0), 1, 7, 0), (14, 355, np.float64(37.0), 1, 4, 1), (15, 153, np.float64(20.0), 1, 5, 1), (16, 75, np.float64(8.0), 1, 1, 0), (17, 129, np.float64(18.0), 1, 2, 1), (18, 1147, np.float64(53.0), 1, 7, 1), (19, 177, np.float64(20.0), 1, 5, 0), (20, 144, np.float64(16.0), 1, 6, 1), (21, 3808, np.float64(76.0), 1, 28, 1), (22, 105, np.float64(12.0), 1, 1, 1), (23, 118, np.float64(13.0), 1, 2, 0), (24, 44, np.float64(5.0), 1, 1, 0), (25, 278, np.float64(32.0), 1, 5, 0), (26, 116, np.float64(14.0), 1, 4, 0), (27, 147, np.float64(17.0), 1, 3, 1), (28, 49, np.float64(4.0), 1, 3, 0), (29, 302, np.float64(30.0), 1, 7, 0), (30, 72, np.float64(6.0), 1, 6, 0)]

=== TABLE ===
Total episodes: 30
Total frames survived: 8742
Total kills: 564.0
Total deaths: 30
K/D Ratio: 18.8
Total items collected: 136
Total weapons collected: 15

Average frames survived: 291.4
Average kills: 18.8
Average deaths: 1.0
Average K/D Ratio: 18.8
Average items collected: 4.533333333333333
Average weapons collected: 0.5


--- GAME FEATURES

uv run levd.py --watch --algorithm rainbow --scenario full_deathmatch --resume-path 812_20251213_200031_1st_Game_Features_Net/policy_best.pth --test_levels 0 --test_maps 4 --use-game-features --game-features "enemy,health,weapon,ammo" --test-num 1000

== EVALUATIONS cig MAP 4 ===
[(1, 72, np.float64(14.0), 1, 2, 1), (2, 125, np.float64(35.0), 1, 3, 1), (3, 32, np.float64(8.0), 1, 3, 3), (4, 239, np.float64(50.0), 1, 4, 1), (5, 175, np.float64(37.0), 1, 5, 1), (6, 65, np.float64(23.0), 1, 2, 2), (7, 101, np.float64(23.0), 1, 2, 1), (8, 34, np.float64(11.0), 1, 1, 1), (9, 35, np.float64(12.0), 1, 2, 1), (10, 62, np.float64(23.0), 1, 2, 1), (11, 158, np.float64(39.0), 1, 6, 2), (12, 67, np.float64(15.0), 1, 1, 1), (13, 93, np.float64(32.0), 1, 5, 2), (14, 100, np.float64(29.0), 1, 3, 1), (15, 96, np.float64(37.0), 1, 4, 1), (16, 210, np.float64(48.0), 1, 4, 1), (17, 90, np.float64(27.0), 1, 3, 1), (18, 56, np.float64(17.0), 1, 1, 1), (19, 124, np.float64(32.0), 1, 2, 1), (20, 316, np.float64(56.0), 1, 7, 1), (21, 142, np.float64(31.0), 1, 4, 1), (22, 261, np.float64(50.0), 1, 5, 1), (23, 358, np.float64(56.0), 1, 5, 1), (24, 43, np.float64(13.0), 1, 2, 1), (25, 181, np.float64(49.0), 1, 4, 1), (26, 53, np.float64(9.0), 1, 2, 1), (27, 85, np.float64(17.0), 1, 3, 2), (28, 102, np.float64(30.0), 1, 3, 1), (29, 124, np.float64(31.0), 1, 1, 1), (30, 253, np.float64(46.0), 1, 5, 1)]

=== TABLE ===
Total episodes: 30
Total frames survived: 3852
Total kills: 900.0
Total deaths: 30
K/D Ratio: 30.0
Total items collected: 96
Total weapons collected: 36

Average frames survived: 128.4
Average kills: 30.0
Average deaths: 1.0
Average K/D Ratio: 30.0
Average items collected: 3.2
Average weapons collected: 1.2

---

=== EVALUATIONS cig MAP 11 ===
[(1, 50, np.float64(3.0), 1, 5, 0), (2, 90, np.float64(13.0), 1, 4, 0), (3, 92, np.float64(13.0), 1, 0, 0), (4, 52, np.float64(5.0), 1, 3, 0), (5, 196, np.float64(22.0), 1, 6, 0), (6, 99, np.float64(11.0), 1, 3, 0), (7, 199, np.float64(23.0), 1, 3, 1), (8, 122, np.float64(14.0), 1, 6, 1), (9, 56, np.float64(4.0), 1, 3, 0), (10, 96, np.float64(15.0), 1, 1, 0), (11, 155, np.float64(20.0), 1, 3, 0), (12, 115, np.float64(15.0), 1, 1, 0), (13, 106, np.float64(16.0), 1, 6, 0), (14, 885, np.float64(55.0), 1, 1, 0), (15, 221, np.float64(22.0), 1, 2, 0), (16, 290, np.float64(28.0), 1, 2, 0), (17, 127, np.float64(13.0), 1, 2, 0), (18, 229, np.float64(28.0), 1, 4, 0), (19, 96, np.float64(11.0), 1, 3, 0), (20, 95, np.float64(12.0), 1, 5, 0), (21, 150, np.float64(18.0), 1, 2, 0), (22, 229, np.float64(25.0), 1, 1, 1), (23, 122, np.float64(15.0), 1, 4, 0), (24, 336, np.float64(32.0), 1, 3, 0), (25, 182, np.float64(18.0), 1, 3, 0), (26, 96, np.float64(14.0), 1, 4, 0), (27, 173, np.float64(18.0), 1, 2, 0), (28, 86, np.float64(19.0), 1, 4, 0), (29, 116, np.float64(13.0), 1, 7, 1), (30, 194, np.float64(23.0), 1, 9, 0)]

=== TABLE ===
Total episodes: 30
Total frames survived: 5055
Total kills: 538.0
Total deaths: 30
K/D Ratio: 17.933333333333334
Total items collected: 102
Total weapons collected: 4

Average frames survived: 168.5
Average kills: 17.933333333333334
Average deaths: 1.0
Average K/D Ratio: 17.933333333333334
Average items collected: 3.4
Average weapons collected: 0.13333333333333333


--- GAME FEATURES + ADAPTIVE MOVEMENT REWARD

uv run levd.py --watch --algorithm rainbow --scenario full_deathmatch --resume-path 812_20251215_230049_AdaptiveMovementRewardWrapper/policy_best.pth --test_levels 0 --test_maps 4 --use-game-features --game-features "enemy,health,weapon,ammo" --test-num 1000

=== EVALUATIONS cig map 4 ===
[(1, 80, np.float64(20.0), 1, 1, 1), (2, 105, np.float64(24.0), 1, 5, 1), (3, 64, np.float64(15.0), 1, 2, 1), (4, 52, np.float64(17.0), 1, 2, 1), (5, 159, np.float64(41.0), 1, 4, 1), (6, 322, np.float64(53.0), 1, 6, 2), (7, 110, np.float64(26.0), 1, 1, 1), (8, 73, np.float64(22.0), 1, 5, 2), (9, 199, np.float64(46.0), 1, 2, 2), (10, 57, np.float64(14.0), 1, 1, 0), (11, 142, np.float64(37.0), 1, 1, 1), (12, 49, np.float64(16.0), 1, 0, 1), (13, 98, np.float64(34.0), 1, 4, 2), (14, 111, np.float64(39.0), 1, 2, 1), (15, 87, np.float64(31.0), 1, 2, 1), (16, 86, np.float64(25.0), 1, 1, 1), (17, 79, np.float64(21.0), 1, 2, 1), (18, 69, np.float64(27.0), 1, 3, 2), (19, 127, np.float64(30.0), 1, 3, 1), (20, 83, np.float64(21.0), 1, 1, 0), (21, 100, np.float64(28.0), 1, 1, 1), (22, 139, np.float64(26.0), 1, 1, 0), (23, 125, np.float64(35.0), 1, 2, 1), (24, 47, np.float64(13.0), 1, 0, 1), (25, 122, np.float64(35.0), 1, 3, 1), (26, 50, np.float64(11.0), 1, 3, 1), (27, 78, np.float64(22.0), 1, 4, 2), (28, 95, np.float64(29.0), 1, 2, 1), (29, 271, np.float64(45.0), 1, 4, 1), (30, 59, np.float64(15.0), 1, 0, 0)]

=== TABLE ===
Total episodes: 30
Total frames survived: 3238
Total kills: 818.0
Total deaths: 30
K/D Ratio: 27.266666666666666
Total items collected: 68
Total weapons collected: 32

Average frames survived: 107.93333333333334
Average kills: 27.266666666666666
Average deaths: 1.0
Average K/D Ratio: 27.266666666666666
Average items collected: 2.2666666666666666
Average weapons collected: 1.0666666666666667

=== EVALUATIONS cig map 11 ===
[(1, 70, np.float64(6.0), 1, 1, 0), (2, 90, np.float64(10.0), 1, 4, 0), (3, 91, np.float64(9.0), 1, 0, 0), (4, 109, np.float64(14.0), 1, 5, 1), (5, 79, np.float64(9.0), 1, 2, 0), (6, 184, np.float64(23.0), 1, 6, 0), (7, 457, np.float64(35.0), 1, 7, 1), (8, 62, np.float64(6.0), 1, 2, 0), (9, 148, np.float64(21.0), 1, 6, 1), (10, 145, np.float64(19.0), 1, 2, 1), (11, 86, np.float64(8.0), 1, 3, 0), (12, 185, np.float64(21.0), 1, 1, 0), (13, 137, np.float64(17.0), 1, 4, 1), (14, 51, np.float64(7.0), 1, 1, 1), (15, 77, np.float64(10.0), 1, 0, 0), (16, 92, np.float64(14.0), 1, 1, 0), (17, 263, np.float64(25.0), 1, 3, 0), (18, 80, np.float64(11.0), 1, 1, 1), (19, 136, np.float64(20.0), 1, 6, 0), (20, 459, np.float64(40.0), 1, 5, 0), (21, 74, np.float64(10.0), 1, 0, 0), (22, 81, np.float64(12.0), 1, 0, 0), (23, 346, np.float64(33.0), 1, 3, 0), (24, 75, np.float64(9.0), 1, 1, 0), (25, 117, np.float64(15.0), 1, 2, 0), (26, 371, np.float64(29.0), 1, 5, 1), (27, 202, np.float64(25.0), 1, 1, 0), (28, 101, np.float64(14.0), 1, 3, 0), (29, 369, np.float64(29.0), 1, 6, 0), (30, 206, np.float64(22.0), 1, 4, 1)]

=== TABLE ===
Total episodes: 30
Total frames survived: 4943
Total kills: 523.0
Total deaths: 30
K/D Ratio: 17.433333333333334
Total items collected: 85
Total weapons collected: 9

Average frames survived: 164.76666666666668
Average kills: 17.433333333333334
Average deaths: 1.0
Average K/D Ratio: 17.433333333333334
Average items collected: 2.8333333333333335
Average weapons collected: 0.3
