from numpy import array
from numpy.ma import masked_array

cv_result_criterion = {
    'mean_fit_time': array([28.20599985, 59.28139911]),
 'std_fit_time': array([0.2386047 , 7.69036074]),
 'mean_score_time': array([2.34799914, 1.43119903]),
 'std_score_time': array([0.07327011, 0.18055123]),
 'param_criterion': masked_array(data=['gini', 'entropy'],
              mask=[False, False],
        fill_value='?',
             dtype=object),
 'params': [{'criterion': 'gini'}, {'criterion': 'entropy'}],
 'split0_test_score': array([-2.50628454, -2.50634108]),
 'split1_test_score': array([-2.51124175, -2.51079595]),
 'split2_test_score': array([-2.50961146, -2.50443974]),
 'split3_test_score': array([-2.5035699, -2.5003022]),
 'split4_test_score': array([-2.50659297, -2.50406911]),
 'mean_test_score': array([-2.5074607 , -2.50519092]),
 'std_test_score': array([0.00269057, 0.00341944]),
 'rank_test_score': array([2, 1]),
 'split0_train_score': array([-2.47758223, -2.4744618 ]),
 'split1_train_score': array([-2.47914046, -2.47449743]),
 'split2_train_score': array([-2.48137667, -2.47406335]),
 'split3_train_score': array([-2.48188648, -2.47579565]),
 'split4_train_score': array([-2.48095416, -2.47633065]),
 'mean_train_score': array([-2.480188  , -2.47502978]),
 'std_train_score': array([0.00159826, 0.00087394])}

cv_result_min_sample_split = {'mean_fit_time': array([68.14419951, 67.56459689, 68.68700128, 67.9132    , 63.88400178]),
 'std_fit_time': array([ 1.16645819,  0.99905091,  1.234406  ,  0.94243593, 10.47608878]),
 'mean_score_time': array([2.83819904, 2.28400211, 2.67139912, 2.26959844, 2.77899876]),
 'std_score_time': array([0.46051421, 0.57398919, 0.52638038, 0.62955865, 0.88756768]),
 'param_min_samples_split': masked_array(data=[20, 40, 60, 80, 100],
              mask=[False, False, False, False, False],
        fill_value='?',
             dtype=object),
 'params': [{'min_samples_split': 20},
  {'min_samples_split': 40},
  {'min_samples_split': 60},
  {'min_samples_split': 80},
  {'min_samples_split': 100}],
 'split0_test_score': array([-2.50775761, -2.50677638, -2.50836831, -2.50660572, -2.50900031]),
 'split1_test_score': array([-2.51174334, -2.51103459, -2.51110268, -2.51113005, -2.51043393]),
 'split2_test_score': array([-2.5041221 , -2.50427744, -2.50455218, -2.50526669, -2.50329724]),
 'split3_test_score': array([-2.50160995, -2.50188272, -2.50109528, -2.50052328, -2.50105069]),
 'split4_test_score': array([-2.50365033, -2.50468034, -2.50283257, -2.50365481, -2.50356638]),
 'mean_test_score': array([-2.50577827, -2.50573145, -2.50559206, -2.50543754, -2.50547151]),
 'std_test_score': array([0.00358193, 0.00307411, 0.00366046, 0.00349647, 0.00360521]),
 'rank_test_score': array([5, 4, 3, 1, 2]),
 'split0_train_score': array([-2.47598819, -2.47483958, -2.47513826, -2.47506173, -2.47636459]),
 'split1_train_score': array([-2.47567093, -2.47499365, -2.47534905, -2.47567231, -2.4753774 ]),
 'split2_train_score': array([-2.47407082, -2.47467922, -2.47536734, -2.47609642, -2.47421015]),
 'split3_train_score': array([-2.47734785, -2.47742325, -2.47736525, -2.47635189, -2.47787261]),
 'split4_train_score': array([-2.47604656, -2.47556853, -2.47523158, -2.47683514, -2.47673104]),
 'mean_train_score': array([-2.47582487, -2.47550084, -2.47569029, -2.4760035 , -2.47611116]),
 'std_train_score': array([0.0010484 , 0.00100696, 0.00084159, 0.00060303, 0.00124124])}

cv_result_max_features = {'mean_fit_time': array([29.6425993 , 34.13300352, 39.65259891, 42.54479885, 45.75400276,
        47.25319629]),
 'std_fit_time': array([0.32565213, 0.78440851, 0.84080009, 0.70716056, 1.24527831,
        1.4623189 ]),
 'mean_score_time': array([2.53760161, 2.78979607, 3.02099867, 2.24359741, 2.43019743,
        1.6618001 ]),
 'std_score_time': array([0.20454355, 0.67267044, 0.14643594, 0.45188484, 0.49079214,
        0.13698278]),
 'param_max_features': masked_array(data=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
              mask=[False, False, False, False, False, False],
        fill_value='?',
             dtype=object),
 'params': [{'max_features': 0.5},
  {'max_features': 0.6},
  {'max_features': 0.7},
  {'max_features': 0.8},
  {'max_features': 0.9},
  {'max_features': 1.0}],
 'split0_test_score': array([-2.50610561, -2.50649544, -2.51486803, -2.51845082, -2.52598927,
        -2.66431618]),
 'split1_test_score': array([-2.50912879, -2.50978106, -2.51236546, -2.51689317, -2.5288907 ,
        -2.64602312]),
 'split2_test_score': array([-2.50858081, -2.51102612, -2.5128911 , -2.51988905, -2.52712523,
        -2.65517438]),
 'split3_test_score': array([-2.50359051, -2.50475116, -2.50716799, -2.51248944, -2.52397748,
        -2.64016278]),
 'split4_test_score': array([-2.5075246 , -2.50768534, -2.51095136, -2.51517029, -2.52301578,
        -2.63083612]),
 'mean_test_score': array([-2.50698628, -2.50794803, -2.51165   , -2.51657956, -2.52580061,
        -2.64730917]),
 'std_test_score': array([0.00198627, 0.00224642, 0.00256845, 0.00257921, 0.00211723,
        0.01162071]),
 'rank_test_score': array([1, 2, 3, 4, 5, 6]),
 'split0_train_score': array([-2.47790496, -2.47720137, -2.48008423, -2.48261401, -2.48503734,
        -2.50266642]),
 'split1_train_score': array([-2.47694125, -2.47736626, -2.47904674, -2.4824011 , -2.4866949 ,
        -2.50266443]),
 'split2_train_score': array([-2.48096191, -2.48267775, -2.48371276, -2.48801426, -2.49156699,
        -2.50874181]),
 'split3_train_score': array([-2.48206032, -2.483241  , -2.48542247, -2.48985181, -2.49508936,
        -2.51285161]),
 'split4_train_score': array([-2.48086861, -2.4817498 , -2.48291115, -2.48745431, -2.49279682,
        -2.51583977]),
 'mean_train_score': array([-2.47974741, -2.48044724, -2.48223547, -2.4860671 , -2.49023708,
        -2.50855281]),
 'std_train_score': array([0.00196727, 0.00262698, 0.00234912, 0.00301341, 0.00378021,
        0.0053092 ])}

cv_result_bootstrap = {'mean_fit_time': array([23.79880614, 27.24199996]),
 'std_fit_time': array([0.4637407, 6.0367362]),
 'mean_score_time': array([2.47599635, 2.53599834]),
 'std_score_time': array([0.10714402, 1.0778334 ]),
 'param_bootstrap': masked_array(data=[True, False],
              mask=[False, False],
        fill_value='?',
             dtype=object),
 'params': [{'bootstrap': True}, {'bootstrap': False}],
 'split0_test_score': array([-2.50747346, -2.50635769]),
 'split1_test_score': array([-2.51041082, -2.51011742]),
 'split2_test_score': array([-2.50743949, -2.50975602]),
 'split3_test_score': array([-2.50383781, -2.50344168]),
 'split4_test_score': array([-2.50456062, -2.50624662]),
 'mean_test_score': array([-2.50674551, -2.50718446]),
 'std_test_score': array([0.00235254, 0.00248126]),
 'rank_test_score': array([1, 2]),
 'split0_train_score': array([-2.47970352, -2.47807732]),
 'split1_train_score': array([-2.47839122, -2.47794565]),
 'split2_train_score': array([-2.47967723, -2.4821203 ]),
 'split3_train_score': array([-2.48249827, -2.48192515]),
 'split4_train_score': array([-2.47916002, -2.48109501]),
 'mean_train_score': array([-2.47988605, -2.48023269]),
 'std_train_score': array([0.00139013, 0.00184647])}

cv_result_max_depth_n_estimators = {'mean_fit_time': array([ 24.65760036,  51.21040149,  78.71460047, 105.93700209,
        133.831004  , 158.75199842, 186.59739947,  30.82140031,
         63.62360497,  89.5189971 , 117.37220125, 145.21100307,
        173.40440149, 206.30680313,  32.41300025,  67.49680061,
         96.57380047, 127.80279846, 160.675002  , 192.21300106,
        231.39140511,  35.37700157,  73.58200049, 106.81480346,
        140.21700144, 175.41700048, 213.65020475, 245.88399696,
         38.55859776,  75.17020183, 112.98800106, 147.98739705,
        185.8455986 , 223.79459829, 233.88320017]),
 'std_fit_time': array([ 0.51771095,  0.8197866 ,  1.51369237,  1.70459985,  2.63219694,
         3.15518078,  3.26365441,  1.31804439,  1.21425327,  1.40346889,
         1.67045245,  1.87636457,  1.72765592,  2.22728743,  0.36628113,
         1.28454483,  1.12478465,  1.25063898,  3.0399443 ,  2.4703355 ,
         2.4857074 ,  0.86948596,  2.3025847 ,  1.1056532 ,  1.88637935,
         2.29648844,  3.46512077,  3.68825761,  0.96968275,  0.3189318 ,
         0.78659636,  1.1286199 ,  1.91663426,  2.08616891, 20.75354716]),
 'mean_score_time': array([ 2.58999996,  4.41340179,  6.57219958,  8.97080226, 11.1677978 ,
        13.2498014 , 16.30280089,  2.52720227,  5.02860055,  6.87799821,
         8.68199649, 10.73420219, 13.00079827, 16.37140055,  2.32059817,
         4.99062719,  7.02060013,  8.78320284, 11.16999722, 13.67499914,
        16.67079544,  2.42259803,  5.37479839,  7.51860046,  8.99520302,
        11.73659816, 14.59659925, 17.702003  ,  2.62479959,  5.24659977,
         7.16160159,  8.95759807, 11.57979946, 14.11800022, 11.71340108]),
 'std_score_time': array([0.12247425, 0.24076884, 0.42578133, 0.58885847, 0.58826309,
        0.44239126, 0.74824981, 0.12360609, 0.28031639, 0.44231807,
        0.61310014, 0.45767994, 0.22614635, 0.49877686, 0.08530231,
        0.29916847, 0.66622517, 0.47055908, 0.41356222, 0.39007953,
        0.57504434, 0.12672109, 0.44955129, 0.59039865, 0.37020469,
        0.5823354 , 0.41199574, 1.215177  , 0.1209063 , 0.35810804,
        0.62172282, 0.29594865, 0.35903874, 0.6327059 , 1.21088744]),
 'param_max_depth': masked_array(data=[6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8,
                    8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10,
                    10],
              mask=[False, False, False, False, False, False, False, False,
                    False, False, False, False, False, False, False, False,
                    False, False, False, False, False, False, False, False,
                    False, False, False, False, False, False, False, False,
                    False, False, False],
        fill_value='?',
             dtype=object),
 'param_n_estimators': masked_array(data=[200, 400, 600, 800, 1000, 1200, 1400, 200, 400, 600,
                    800, 1000, 1200, 1400, 200, 400, 600, 800, 1000, 1200,
                    1400, 200, 400, 600, 800, 1000, 1200, 1400, 200, 400,
                    600, 800, 1000, 1200, 1400],
              mask=[False, False, False, False, False, False, False, False,
                    False, False, False, False, False, False, False, False,
                    False, False, False, False, False, False, False, False,
                    False, False, False, False, False, False, False, False,
                    False, False, False],
        fill_value='?',
             dtype=object),
 'params': [{'max_depth': 6, 'n_estimators': 200},
  {'max_depth': 6, 'n_estimators': 400},
  {'max_depth': 6, 'n_estimators': 600},
  {'max_depth': 6, 'n_estimators': 800},
  {'max_depth': 6, 'n_estimators': 1000},
  {'max_depth': 6, 'n_estimators': 1200},
  {'max_depth': 6, 'n_estimators': 1400},
  {'max_depth': 7, 'n_estimators': 200},
  {'max_depth': 7, 'n_estimators': 400},
  {'max_depth': 7, 'n_estimators': 600},
  {'max_depth': 7, 'n_estimators': 800},
  {'max_depth': 7, 'n_estimators': 1000},
  {'max_depth': 7, 'n_estimators': 1200},
  {'max_depth': 7, 'n_estimators': 1400},
  {'max_depth': 8, 'n_estimators': 200},
  {'max_depth': 8, 'n_estimators': 400},
  {'max_depth': 8, 'n_estimators': 600},
  {'max_depth': 8, 'n_estimators': 800},
  {'max_depth': 8, 'n_estimators': 1000},
  {'max_depth': 8, 'n_estimators': 1200},
  {'max_depth': 8, 'n_estimators': 1400},
  {'max_depth': 9, 'n_estimators': 200},
  {'max_depth': 9, 'n_estimators': 400},
  {'max_depth': 9, 'n_estimators': 600},
  {'max_depth': 9, 'n_estimators': 800},
  {'max_depth': 9, 'n_estimators': 1000},
  {'max_depth': 9, 'n_estimators': 1200},
  {'max_depth': 9, 'n_estimators': 1400},
  {'max_depth': 10, 'n_estimators': 200},
  {'max_depth': 10, 'n_estimators': 400},
  {'max_depth': 10, 'n_estimators': 600},
  {'max_depth': 10, 'n_estimators': 800},
  {'max_depth': 10, 'n_estimators': 1000},
  {'max_depth': 10, 'n_estimators': 1200},
  {'max_depth': 10, 'n_estimators': 1400}],
 'split0_test_score': array([-2.50667651, -2.5071556 , -2.50702043, -2.50680655, -2.50732225,
        -2.50680289, -2.50692638, -2.48647589, -2.48642363, -2.48710644,
        -2.48666688, -2.48685003, -2.48691802, -2.48697884, -2.47417324,
        -2.47111955, -2.47073992, -2.47051122, -2.47026415, -2.46893024,
        -2.46953315, -2.45965948, -2.45888193, -2.45603989, -2.45644936,
        -2.45613578, -2.45613741, -2.45617582, -2.45353306, -2.44857967,
        -2.44773109, -2.4474725 , -2.44457056, -2.44463782, -2.44494118]),
 'split1_test_score': array([-2.51140771, -2.5110845 , -2.51198711, -2.51181521, -2.5104484 ,
        -2.5115685 , -2.51139704, -2.49193265, -2.49078648, -2.49004887,
        -2.49007804, -2.49021001, -2.48936022, -2.48998447, -2.47117432,
        -2.47139231, -2.47195339, -2.47120383, -2.47142409, -2.47144971,
        -2.47097235, -2.45761524, -2.45608566, -2.45590916, -2.45677616,
        -2.45646744, -2.45597588, -2.4559923 , -2.44751312, -2.44677144,
        -2.44719175, -2.44439551, -2.44393039, -2.44419023, -2.44399649]),
 'split2_test_score': array([-2.50831134, -2.5082798 , -2.50753903, -2.50752146, -2.50777143,
        -2.50770661, -2.50755559, -2.48846389, -2.48788367, -2.48781789,
        -2.48776946, -2.48781383, -2.48765204, -2.48766482, -2.47024582,
        -2.46994447, -2.47060088, -2.47035056, -2.47007309, -2.47023259,
        -2.470136  , -2.45895459, -2.45621655, -2.45618515, -2.45577546,
        -2.4553676 , -2.4556465 , -2.45553734, -2.44667281, -2.44663901,
        -2.44504139, -2.44401182, -2.44424068, -2.44394474, -2.44401004]),
 'split3_test_score': array([-2.50415989, -2.50382922, -2.50373781, -2.50415753, -2.50386122,
        -2.50358666, -2.50410519, -2.48331339, -2.48307266, -2.48265315,
        -2.48264922, -2.48300069, -2.48283628, -2.48245825, -2.46371894,
        -2.46436073, -2.46365548, -2.46340847, -2.46367461, -2.46346864,
        -2.46328669, -2.45486412, -2.45030786, -2.44800443, -2.44750049,
        -2.44753762, -2.44762351, -2.44810291, -2.44168029, -2.43504381,
        -2.43667848, -2.43556894, -2.43627946, -2.43513396, -2.4358013 ]),
 'split4_test_score': array([-2.50504968, -2.50531415, -2.50502111, -2.50530285, -2.50544086,
        -2.50488185, -2.50532765, -2.48835389, -2.48653386, -2.4852117 ,
        -2.48414461, -2.48589019, -2.48575496, -2.48456839, -2.4709419 ,
        -2.46906609, -2.46760142, -2.46723756, -2.46753881, -2.46878856,
        -2.46858189, -2.45314918, -2.45561145, -2.45242289, -2.45494375,
        -2.45347049, -2.45380069, -2.45392969, -2.44605   , -2.44366657,
        -2.4426648 , -2.44319891, -2.44162593, -2.44197717, -2.44256282]),
 'mean_test_score': array([-2.5071219 , -2.50713359, -2.50706213, -2.50712161, -2.50696972,
        -2.5069103 , -2.50706326, -2.48770833, -2.48694069, -2.48656858,
        -2.48626272, -2.48675373, -2.48650508, -2.48633202, -2.47005216,
        -2.46917764, -2.46891152, -2.46854361, -2.46859613, -2.46857468,
        -2.46850287, -2.45684994, -2.45542185, -2.45371367, -2.45429017,
        -2.45379708, -2.45383799, -2.45394874, -2.4470918 , -2.44414208,
        -2.44386338, -2.44293116, -2.44213064, -2.4419781 , -2.44226356]),
 'std_test_score': array([0.00257093, 0.00249605, 0.00281787, 0.00262238, 0.0022301 ,
        0.00273881, 0.00248217, 0.00281659, 0.0024929 , 0.00249521,
        0.00262792, 0.00236251, 0.00217519, 0.00259413, 0.00344031,
        0.00254847, 0.00299281, 0.00290797, 0.00276745, 0.00272944,
        0.00272148, 0.00247151, 0.00280116, 0.00318093, 0.00345138,
        0.00329678, 0.00321679, 0.00302734, 0.003801  , 0.00481306,
        0.00401192, 0.00395487, 0.00310246, 0.00354018, 0.00331848]),
 'rank_test_score': array([34, 35, 31, 33, 30, 29, 32, 28, 27, 25, 22, 26, 24, 23, 21, 20, 19,
        16, 18, 17, 15, 14, 13,  8, 12,  9, 10, 11,  7,  6,  5,  4,  2,  1,
         3]),
 'split0_train_score': array([-2.47886404, -2.47948716, -2.47970092, -2.47941044, -2.48018939,
        -2.4794713 , -2.47963596, -2.43940644, -2.44007481, -2.44109489,
        -2.44048984, -2.44070485, -2.44113154, -2.44091658, -2.3979039 ,
        -2.39706462, -2.39684326, -2.3963637 , -2.39646669, -2.39628881,
        -2.39678974, -2.34937961, -2.34747215, -2.34866739, -2.34845719,
        -2.34802735, -2.34838237, -2.34840664, -2.29917242, -2.2979863 ,
        -2.29780854, -2.29885627, -2.29828434, -2.29768701, -2.29826634]),
 'split1_train_score': array([-2.48029464, -2.47994697, -2.48072253, -2.48031896, -2.48010089,
        -2.48009192, -2.47995657, -2.44312686, -2.44194573, -2.44105601,
        -2.44080454, -2.44094024, -2.44169165, -2.44096383, -2.39618419,
        -2.39702493, -2.39730809, -2.39707395, -2.39673334, -2.39668637,
        -2.39634558, -2.34893356, -2.34900834, -2.34834385, -2.34965274,
        -2.3489515 , -2.34843066, -2.34851267, -2.29880781, -2.29819993,
        -2.2989888 , -2.29845554, -2.29813619, -2.29892034, -2.29806655]),
 'split2_train_score': array([-2.48221504, -2.48123581, -2.48069769, -2.48068468, -2.4808381 ,
        -2.48074813, -2.48079082, -2.44274661, -2.44244171, -2.44201183,
        -2.4421302 , -2.44217983, -2.44190334, -2.44198107, -2.39784952,
        -2.39834057, -2.39877154, -2.39787865, -2.39842512, -2.39810641,
        -2.3980648 , -2.34949702, -2.35065172, -2.35122121, -2.35117722,
        -2.34992418, -2.3496228 , -2.35074925, -2.30156979, -2.30129504,
        -2.29964241, -2.30021309, -2.29996386, -2.30031103, -2.30019452]),
 'split3_train_score': array([-2.48342414, -2.48304864, -2.4832174 , -2.48342938, -2.48315002,
        -2.48290211, -2.48333888, -2.44564874, -2.44519149, -2.44521618,
        -2.44547126, -2.44569605, -2.44558291, -2.44516794, -2.40142469,
        -2.4021693 , -2.40223558, -2.40182759, -2.40207126, -2.40200224,
        -2.40159367, -2.35580443, -2.35354986, -2.35458351, -2.35355625,
        -2.35375365, -2.35370735, -2.35392519, -2.30572638, -2.30267072,
        -2.30382405, -2.30428259, -2.30265216, -2.30320356, -2.30218506]),
 'split4_train_score': array([-2.479844  , -2.48050561, -2.48018964, -2.48059974, -2.48040622,
        -2.47997312, -2.48063162, -2.44071744, -2.4416287 , -2.44192095,
        -2.44040066, -2.44128593, -2.44088103, -2.44099289, -2.39734461,
        -2.39685043, -2.39619539, -2.39578708, -2.39629078, -2.39648066,
        -2.39635908, -2.34758153, -2.34772683, -2.34652326, -2.34713413,
        -2.3473781 , -2.34785799, -2.34796429, -2.29708268, -2.29542509,
        -2.29633808, -2.29726322, -2.29691303, -2.29749327, -2.2958609 ]),
 'mean_train_score': array([-2.48092837, -2.48084484, -2.48090564, -2.48088864, -2.48093692,
        -2.48063732, -2.48087077, -2.44232922, -2.44225649, -2.44225997,
        -2.4418593 , -2.44216138, -2.44223809, -2.44200446, -2.39814138,
        -2.39828997, -2.39827077, -2.39778619, -2.39799744, -2.3979129 ,
        -2.39783057, -2.35023923, -2.34968178, -2.34986785, -2.34999551,
        -2.34960696, -2.34960023, -2.34991161, -2.30047182, -2.29911542,
        -2.29932038, -2.29981414, -2.29918992, -2.29952304, -2.29891468]),
 'std_train_score': array([0.0016567 , 0.00124693, 0.00121532, 0.00134834, 0.00113551,
        0.00120331, 0.00130504, 0.00214337, 0.00166727, 0.00153118,
        0.00190972, 0.00183702, 0.00171257, 0.00163082, 0.00175426,
        0.00201126, 0.00215602, 0.00213863, 0.00217406, 0.00214259,
        0.00198341, 0.00286459, 0.00223878, 0.00279353, 0.00222471,
        0.00224447, 0.00213333, 0.00222821, 0.00299198, 0.00257381,
        0.002517  , 0.00242474, 0.00198498, 0.00209864, 0.00213535])}