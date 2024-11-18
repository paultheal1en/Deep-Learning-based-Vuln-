int main(void)

{

    int x = 0;

    int i, j;

    AVLFG state;



    av_lfg_init(&state, 0xdeadbeef);

    for (j = 0; j < 10000; j++) {

        START_TIMER

        for (i = 0; i < 624; i++) {

            //av_log(NULL, AV_LOG_ERROR, "%X\n", av_lfg_get(&state));

            x += av_lfg_get(&state);

        }

        STOP_TIMER("624 calls of av_lfg_get");

    }

    av_log(NULL, AV_LOG_ERROR, "final value:%X\n", x);



    /* BMG usage example */

    {

        double mean   = 1000;

        double stddev = 53;



        av_lfg_init(&state, 42);



        for (i = 0; i < 1000; i += 2) {

            double bmg_out[2];

            av_bmg_get(&state, bmg_out);

            av_log(NULL, AV_LOG_INFO,

                   "%f\n%f\n",

                   bmg_out[0] * stddev + mean,

                   bmg_out[1] * stddev + mean);

        }

    }



    return 0;

}
