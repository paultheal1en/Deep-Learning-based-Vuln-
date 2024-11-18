static int check(AVIOContext *pb, int64_t pos, int64_t *out_pos)

{

    MPADecodeHeader mh = { 0 };

    int i;

    uint32_t header;

    int64_t off = 0;





    for (i = 0; i < SEEK_PACKETS; i++) {

        off = avio_seek(pb, pos + mh.frame_size, SEEK_SET);

        if (off < 0)

            break;



        header = avio_rb32(pb);



        if (ff_mpa_check_header(header) < 0 ||

            avpriv_mpegaudio_decode_header(&mh, header))

            break;

        out_pos[i] = off;

    }



    return i;

}
