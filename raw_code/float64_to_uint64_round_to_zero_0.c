uint64_t float64_to_uint64_round_to_zero (float64 a STATUS_PARAM)

{

    int64_t v;



    v = int64_to_float64(INT64_MIN STATUS_VAR);

    v = float64_to_int64_round_to_zero((a + v) STATUS_VAR);



    return v - INT64_MIN;

}
