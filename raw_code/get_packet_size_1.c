static int get_packet_size(const uint8_t *buf, int size)

{

    int score, fec_score, dvhs_score;



    if (size < (TS_FEC_PACKET_SIZE * 5 + 1))

        return AVERROR_INVALIDDATA;



    score      = analyze(buf, size, TS_PACKET_SIZE, NULL);

    dvhs_score = analyze(buf, size, TS_DVHS_PACKET_SIZE, NULL);

    fec_score  = analyze(buf, size, TS_FEC_PACKET_SIZE, NULL);

    av_dlog(NULL, "score: %d, dvhs_score: %d, fec_score: %d \n",

            score, dvhs_score, fec_score);



    if (score > fec_score && score > dvhs_score)

        return TS_PACKET_SIZE;

    else if (dvhs_score > score && dvhs_score > fec_score)

        return TS_DVHS_PACKET_SIZE;

    else if (score < fec_score && dvhs_score < fec_score)

        return TS_FEC_PACKET_SIZE;

    else

        return AVERROR_INVALIDDATA;

}
