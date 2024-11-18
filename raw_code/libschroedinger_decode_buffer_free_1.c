static void libschroedinger_decode_buffer_free(SchroBuffer *schro_buf,

                                               void *priv)

{

    av_freep(&priv);

}
