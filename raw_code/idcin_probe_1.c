static int idcin_probe(AVProbeData *p)

{

    unsigned int number, sample_rate;



    /*

     * This is what you could call a "probabilistic" file check: id CIN

     * files don't have a definite file signature. In lieu of such a marker,

     * perform sanity checks on the 5 32-bit header fields:

     *  width, height: greater than 0, less than or equal to 1024

     * audio sample rate: greater than or equal to 8000, less than or

     *  equal to 48000, or 0 for no audio

     * audio sample width (bytes/sample): 0 for no audio, or 1 or 2

     * audio channels: 0 for no audio, or 1 or 2

     */



    /* check we have enough data to do all checks, otherwise the

       0-padding may cause a wrong recognition */

    if (p->buf_size < 20)

        return 0;



    /* check the video width */

    number = AV_RL32(&p->buf[0]);

    if ((number == 0) || (number > 1024))

       return 0;



    /* check the video height */

    number = AV_RL32(&p->buf[4]);

    if ((number == 0) || (number > 1024))

       return 0;



    /* check the audio sample rate */

    sample_rate = AV_RL32(&p->buf[8]);

    if (sample_rate && (sample_rate < 8000 || sample_rate > 48000))

        return 0;



    /* check the audio bytes/sample */

    number = AV_RL32(&p->buf[12]);

    if (number > 2 || sample_rate && !number)

        return 0;



    /* check the audio channels */

    number = AV_RL32(&p->buf[16]);

    if (number > 2 || sample_rate && !number)

        return 0;



    /* return half certainty since this check is a bit sketchy */

    return AVPROBE_SCORE_EXTENSION;

}
