uint32_t lm4549_write_samples(lm4549_state *s, uint32_t left, uint32_t right)

{

    /* The left and right samples are in 20-bit resolution.

       The LM4549 has 18-bit resolution and only uses the bits [19:2].

       This model supports 16-bit playback.

    */



    if (s->buffer_level >= LM4549_BUFFER_SIZE) {

        DPRINTF("write_sample Buffer full\n");

        return 0;

    }



    /* Store 16-bit samples in the buffer */

    s->buffer[s->buffer_level++] = (left >> 4);

    s->buffer[s->buffer_level++] = (right >> 4);



    if (s->buffer_level == LM4549_BUFFER_SIZE) {

        /* Trigger the transfer of the buffer to the audio host */

        lm4549_audio_transfer(s);

    }



    return 1;

}
