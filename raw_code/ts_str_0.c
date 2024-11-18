static void ts_str(char buffer[60], int64_t ts, AVRational base)

{

    if (ts == AV_NOPTS_VALUE) {

        strcpy(buffer, " NOPTS   ");

        return;

    }

    ts= av_rescale_q(ts, base, (AVRational){1, 1000000});

    snprintf(buffer, 60, "%c%Ld.%06Ld", ts<0 ? '-' : ' ', FFABS(ts)/1000000, FFABS(ts)%1000000);

}
