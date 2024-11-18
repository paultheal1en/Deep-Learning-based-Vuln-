void ff_update_duplicate_context(MpegEncContext *dst, MpegEncContext *src)

{

    MpegEncContext bak;

    int i;

    // FIXME copy only needed parts

    // START_TIMER

    backup_duplicate_context(&bak, dst);

    memcpy(dst, src, sizeof(MpegEncContext));

    backup_duplicate_context(dst, &bak);

    for (i = 0; i < 12; i++) {

        dst->pblocks[i] = &dst->block[i];

    }

    // STOP_TIMER("update_duplicate_context")

    // about 10k cycles / 0.01 sec for  1000frames on 1ghz with 2 threads

}
