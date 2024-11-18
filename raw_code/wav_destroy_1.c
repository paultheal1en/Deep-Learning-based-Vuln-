static void wav_destroy (void *opaque)

{

    WAVState *wav = opaque;

    uint8_t rlen[4];

    uint8_t dlen[4];

    uint32_t datalen = wav->bytes;

    uint32_t rifflen = datalen + 36;



    if (!wav->f) {

        return;

    }



    le_store (rlen, rifflen, 4);

    le_store (dlen, datalen, 4);



    qemu_fseek (wav->f, 4, SEEK_SET);

    qemu_put_buffer (wav->f, rlen, 4);



    qemu_fseek (wav->f, 32, SEEK_CUR);

    qemu_put_buffer (wav->f, dlen, 4);

    qemu_fclose (wav->f);

    if (wav->path) {

        qemu_free (wav->path);

    }

}
