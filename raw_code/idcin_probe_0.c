static int idcin_probe(AVProbeData *p)

{

    unsigned int number;



    /*

     * This is what you could call a "probabilistic" file check: Id CIN

     * files don't have a definite file signature. In lieu of such a marker,

     * perform sanity checks on the 5 32-bit header fields:

     *  width, height: greater than 0, less than or equal to 1024

     * audio sample rate: greater than or equal to 8000, less than or

     *  equal to 48000, or 0 for no audio

     * audio sample width (bytes/sample): 0 for no audio, or 1 or 2

     * audio channels: 0 for no audio, or 1 or 2

     */



    /* cannot proceed without 20 bytes */

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

    number = AV_RL32(&p->buf[8]);

    if ((number != 0) && ((number < 8000) | (number > 48000)))

        return 0;



    /* check the audio bytes/sample */

    number = AV_RL32(&p->buf[12]);

    if (number > 2)

        return 0;



    /* check the audio channels */

    number = AV_RL32(&p->buf[16]);

    if (number > 2)

        return 0;



    /* return half certainly since this check is a bit sketchy */

    return AVPROBE_SCORE_MAX / 2;

}
