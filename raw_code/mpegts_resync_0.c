static int mpegts_resync(ByteIOContext *pb)

{

    int c, i;



    for(i = 0;i < MAX_RESYNC_SIZE; i++) {

        c = url_fgetc(pb);

        if (c < 0)

            return -1;

        if (c == 0x47) {

            url_fseek(pb, -1, SEEK_CUR);

            return 0;

        }

    }

    /* no sync found */

    return -1;

}
