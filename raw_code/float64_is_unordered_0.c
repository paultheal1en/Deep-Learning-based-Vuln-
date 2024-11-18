static int float64_is_unordered(int sig, float64 a, float64 b STATUS_PARAM)

{

    if (float64_is_signaling_nan(a) ||

        float64_is_signaling_nan(b) ||

        (sig && (float64_is_nan(a) || float64_is_nan(b)))) {

        float_raise(float_flag_invalid, status);

        return 1;

    } else if (float64_is_nan(a) || float64_is_nan(b)) {

        return 1;

    } else {

        return 0;

    }

}
