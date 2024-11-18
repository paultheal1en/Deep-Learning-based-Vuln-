static int mov_read_tkhd(MOVContext *c, ByteIOContext *pb, MOVAtom atom)

{

    int i;

    int width;

    int height;

    int64_t disp_transform[2];

    int display_matrix[3][2];

    AVStream *st = c->fc->streams[c->fc->nb_streams-1];

    MOVStreamContext *sc = st->priv_data;

    int version = get_byte(pb);



    get_be24(pb); /* flags */

    /*

    MOV_TRACK_ENABLED 0x0001

    MOV_TRACK_IN_MOVIE 0x0002

    MOV_TRACK_IN_PREVIEW 0x0004

    MOV_TRACK_IN_POSTER 0x0008

    */



    if (version == 1) {

        get_be64(pb);

        get_be64(pb);

    } else {

        get_be32(pb); /* creation time */

        get_be32(pb); /* modification time */

    }

    st->id = (int)get_be32(pb); /* track id (NOT 0 !)*/

    get_be32(pb); /* reserved */



    /* highlevel (considering edits) duration in movie timebase */

    (version == 1) ? get_be64(pb) : get_be32(pb);

    get_be32(pb); /* reserved */

    get_be32(pb); /* reserved */



    get_be16(pb); /* layer */

    get_be16(pb); /* alternate group */

    get_be16(pb); /* volume */

    get_be16(pb); /* reserved */



    //read in the display matrix (outlined in ISO 14496-12, Section 6.2.2)

    // they're kept in fixed point format through all calculations

    // ignore u,v,z b/c we don't need the scale factor to calc aspect ratio

    for (i = 0; i < 3; i++) {

        display_matrix[i][0] = get_be32(pb);   // 16.16 fixed point

        display_matrix[i][1] = get_be32(pb);   // 16.16 fixed point

        get_be32(pb);           // 2.30 fixed point (not used)

    }



    width = get_be32(pb);       // 16.16 fixed point track width

    height = get_be32(pb);      // 16.16 fixed point track height

    sc->width = width >> 16;

    sc->height = height >> 16;



    //transform the display width/height according to the matrix

    // skip this if the display matrix is the default identity matrix

    // to keep the same scale, use [width height 1<<16]

    if (width && height &&

        (display_matrix[0][0] != 65536 || display_matrix[0][1]           ||

        display_matrix[1][0]           || display_matrix[1][1] != 65536  ||

        display_matrix[2][0]           || display_matrix[2][1])) {

        for (i = 0; i < 2; i++)

            disp_transform[i] =

                (int64_t)  width  * display_matrix[0][i] +

                (int64_t)  height * display_matrix[1][i] +

                ((int64_t) display_matrix[2][i] << 16);



        //sample aspect ratio is new width/height divided by old width/height

        st->sample_aspect_ratio = av_d2q(

            ((double) disp_transform[0] * height) /

            ((double) disp_transform[1] * width), INT_MAX);

    }

    return 0;

}
