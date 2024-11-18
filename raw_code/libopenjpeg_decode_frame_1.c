static int libopenjpeg_decode_frame(AVCodecContext *avctx,

                                    void *data, int *data_size,

                                    AVPacket *avpkt)

{

    uint8_t *buf = avpkt->data;

    int buf_size = avpkt->size;

    LibOpenJPEGContext *ctx = avctx->priv_data;

    AVFrame *picture = &ctx->image, *output = data;

    opj_dinfo_t *dec;

    opj_cio_t *stream;

    opj_image_t *image;

    int width, height, ret = -1;

    int pixel_size = 0;

    int ispacked = 0;



    *data_size = 0;



    // Check if input is a raw jpeg2k codestream or in jp2 wrapping

    if((AV_RB32(buf) == 12) &&

       (AV_RB32(buf + 4) == JP2_SIG_TYPE) &&

       (AV_RB32(buf + 8) == JP2_SIG_VALUE)) {

        dec = opj_create_decompress(CODEC_JP2);

    } else {

        // If the AVPacket contains a jp2c box, then skip to

        // the starting byte of the codestream.

        if (AV_RB32(buf + 4) == AV_RB32("jp2c"))

            buf += 8;

        dec = opj_create_decompress(CODEC_J2K);




    if(!dec) {

        av_log(avctx, AV_LOG_ERROR, "Error initializing decoder.\n");



    opj_set_event_mgr((opj_common_ptr)dec, NULL, NULL);



    ctx->dec_params.cp_limit_decoding = LIMIT_TO_MAIN_HEADER;

    // Tie decoder with decoding parameters

    opj_setup_decoder(dec, &ctx->dec_params);

    stream = opj_cio_open((opj_common_ptr)dec, buf, buf_size);

    if(!stream) {

        av_log(avctx, AV_LOG_ERROR, "Codestream could not be opened for reading.\n");






    // Decode the header only

    image = opj_decode_with_info(dec, stream, NULL);

    opj_cio_close(stream);






    width  = image->x1 - image->x0;

    height = image->y1 - image->y0;

    if(av_image_check_size(width, height, 0, avctx) < 0) {

        av_log(avctx, AV_LOG_ERROR, "%dx%d dimension invalid.\n", width, height);

        goto done;


    avcodec_set_dimensions(avctx, width, height);



    switch (image->numcomps) {

    case 1:  avctx->pix_fmt = (image->comps[0].bpp == 8) ? PIX_FMT_GRAY8 : PIX_FMT_GRAY16;

             break;

    case 2:  avctx->pix_fmt = PIX_FMT_GRAY8A;

             break;

    case 3:

    case 4:  avctx->pix_fmt = check_image_attributes(avctx, image);

             break;

    default: av_log(avctx, AV_LOG_ERROR, "%d components unsupported.\n", image->numcomps);

             goto done;




    if(picture->data[0])

        ff_thread_release_buffer(avctx, picture);



    if(ff_thread_get_buffer(avctx, picture) < 0){

        av_log(avctx, AV_LOG_ERROR, "ff_thread_get_buffer() failed\n");





    ctx->dec_params.cp_limit_decoding = NO_LIMITATION;

    ctx->dec_params.cp_reduce = avctx->lowres;

    // Tie decoder with decoding parameters

    opj_setup_decoder(dec, &ctx->dec_params);

    stream = opj_cio_open((opj_common_ptr)dec, buf, buf_size);

    if(!stream) {

        av_log(avctx, AV_LOG_ERROR, "Codestream could not be opened for reading.\n");






    // Decode the codestream

    image = opj_decode_with_info(dec, stream, NULL);

    opj_cio_close(stream);








    pixel_size = av_pix_fmt_descriptors[avctx->pix_fmt].comp[0].step_minus1 + 1;

    ispacked = libopenjpeg_ispacked(avctx->pix_fmt);



    switch (pixel_size) {

    case 1:

        if (ispacked) {

            libopenjpeg_copy_to_packed8(picture, image);

        } else {

            libopenjpeg_copyto8(picture, image);


        break;

    case 2:

        if (ispacked) {

            libopenjpeg_copy_to_packed8(picture, image);

        } else {

            libopenjpeg_copyto16(picture, image);


        break;

    case 3:

    case 4:

        if (ispacked) {

            libopenjpeg_copy_to_packed8(picture, image);


        break;

    case 6:

    case 8:

        if (ispacked) {

            libopenjpeg_copy_to_packed16(picture, image);


        break;

    default:

        av_log(avctx, AV_LOG_ERROR, "unsupported pixel size %d\n", pixel_size);

        goto done;




    *output    = ctx->image;

    *data_size = sizeof(AVPicture);

    ret = buf_size;



done:

    opj_image_destroy(image);


    return ret;
