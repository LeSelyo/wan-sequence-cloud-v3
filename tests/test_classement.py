from scripts import classement as cl


def test_verdict_rules_follow_what_the_user_said():
    assert cl.verdict_of("phaseK/k_craters.mp4")[0] == "R" and cl.verdict_of("phaseK/k_craters.mp4")[1] == cl.USER
    assert cl.verdict_of("phaseK/k_multi.mp4")[0] == "E" and cl.verdict_of("phaseK/k_pencil_general.mp4")[0] == "E"
    assert cl.verdict_of("phaseF/e_table_enhance_gurren_18_1.mp4")[0] == "E" and cl.verdict_of("phaseF/e2_benchcat_sleep.mp4")[0] == "E"
    assert cl.verdict_of("phaseF/e_train_enhance_gurren_18_1.mp4")[0] == "R"
    assert cl.verdict_of("phaseT/t_union_s2.mp4")[0] == "E" and cl.verdict_of("phaseT/t_union_s3.mp4")[0] == "E"
    assert cl.verdict_of("phaseT/t_union_s1.mp4")[0] == "N"  # the user only said seeds 2 and 3
    assert cl.verdict_of("video_runs/trend_7/trend_7.mp4")[0] == "R"
    assert cl.verdict_of("phaseG/g_calm_a.png")[0] == "R" and cl.verdict_of("phaseG/g_calm_a.mp4")[0] == "E"
    assert cl.verdict_of("phaseB/s01_enhance_49f.mp4")[0] == "N"  # nothing said: not judged


def test_the_rain_chain_is_a_failure_but_the_rainy_street_is_not_caught_by_the_same_rule():
    for failed in ("phaseS/s_rain_chain.mp4", "phaseS/s_rain_chain_smooth.mp4", "phaseS/s_rain_K0_K1.mp4", "phaseS/s_rain_k2.png"):
        assert cl.verdict_of(failed)[0] == "E", failed
    for kept in ("phaseS/s_rainy_street.mp4", "phaseS/s_rainy_street_crowd.mp4", "phaseS/s_station_crowd.mp4", "phaseS/s_valley.mp4"):
        assert cl.verdict_of(kept)[0] == "R", kept


def test_every_file_has_a_unique_flat_name_and_a_verdict():
    files = cl.media_files()
    assert files
    names = [cl.flat_name(key) for key, _ in files]
    assert len(names) == len(set(names))
    assert all(cl.verdict_of(key)[0] in cl.FOLDERS for key, _ in files)
