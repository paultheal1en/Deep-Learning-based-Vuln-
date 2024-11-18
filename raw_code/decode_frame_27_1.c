static int decode_frame(AVCodecContext *avctx,
                        void *data, int *got_frame,
                        AVPacket *avpkt)
{
    SheerVideoContext *s = avctx->priv_data;
    ThreadFrame frame = { .f = data };
    AVFrame *p = data;
    GetBitContext gb;
    unsigned format;
    int ret;
    if (avpkt->size <= 20)
    if (AV_RL32(avpkt->data) != MKTAG('S','h','i','r') &&
        AV_RL32(avpkt->data) != MKTAG('Z','w','a','k'))
    s->alt = 0;
    format = AV_RL32(avpkt->data + 16);
    av_log(avctx, AV_LOG_DEBUG, "format: %s\n", av_fourcc2str(format));
    switch (format) {
    case MKTAG(' ', 'R', 'G', 'B'):
        avctx->pix_fmt = AV_PIX_FMT_RGB0;
        s->decode_frame = decode_rgb;
        if (s->format != format) {
            ret  = build_vlc(&s->vlc[0], l_r_rgb, 256);
            ret |= build_vlc(&s->vlc[1], l_g_rgb, 256);
        break;
    case MKTAG(' ', 'r', 'G', 'B'):
        avctx->pix_fmt = AV_PIX_FMT_RGB0;
        s->decode_frame = decode_rgbi;
        if (s->format != format) {
            ret  = build_vlc(&s->vlc[0], l_r_rgbi, 256);
            ret |= build_vlc(&s->vlc[1], l_g_rgbi, 256);
        break;
    case MKTAG('A', 'R', 'G', 'X'):
        avctx->pix_fmt = AV_PIX_FMT_GBRAP10;
        s->decode_frame = decode_argx;
        if (s->format != format) {
            ret  = build_vlc(&s->vlc[0], l_r_rgbx, 1024);
            ret |= build_vlc(&s->vlc[1], l_g_rgbx, 1024);
        break;
    case MKTAG('A', 'r', 'G', 'X'):
        avctx->pix_fmt = AV_PIX_FMT_GBRAP10;
        s->decode_frame = decode_argxi;
        if (s->format != format) {
            ret  = build_vlc(&s->vlc[0], l_r_rgbxi, 1024);
            ret |= build_vlc(&s->vlc[1], l_g_rgbxi, 1024);
        break;
    case MKTAG('R', 'G', 'B', 'X'):
        avctx->pix_fmt = AV_PIX_FMT_GBRP10;
        s->decode_frame = decode_rgbx;
        if (s->format != format) {
            ret  = build_vlc(&s->vlc[0], l_r_rgbx, 1024);
            ret |= build_vlc(&s->vlc[1], l_g_rgbx, 1024);
        break;
    case MKTAG('r', 'G', 'B', 'X'):
        avctx->pix_fmt = AV_PIX_FMT_GBRP10;
        s->decode_frame = decode_rgbxi;
        if (s->format != format) {
            ret  = build_vlc(&s->vlc[0], l_r_rgbxi, 1024);
            ret |= build_vlc(&s->vlc[1], l_g_rgbxi, 1024);
        break;
    case MKTAG('A', 'R', 'G', 'B'):
        avctx->pix_fmt = AV_PIX_FMT_ARGB;
        s->decode_frame = decode_argb;
        if (s->format != format) {
            ret  = build_vlc(&s->vlc[0], l_r_rgb, 256);
            ret |= build_vlc(&s->vlc[1], l_g_rgb, 256);
        break;
    case MKTAG('A', 'r', 'G', 'B'):
        avctx->pix_fmt = AV_PIX_FMT_ARGB;
        s->decode_frame = decode_argbi;
        if (s->format != format) {
            ret  = build_vlc(&s->vlc[0], l_r_rgbi, 256);
            ret |= build_vlc(&s->vlc[1], l_g_rgbi, 256);
        break;
    case MKTAG('A', 'Y', 'B', 'R'):
        s->alt = 1;
    case MKTAG('A', 'Y', 'b', 'R'):
        avctx->pix_fmt = AV_PIX_FMT_YUVA444P;
        s->decode_frame = decode_aybr;
        if (s->format != format) {
            ret  = build_vlc(&s->vlc[0], l_y_ybr, 256);
            ret |= build_vlc(&s->vlc[1], l_u_ybr, 256);
        break;
    case MKTAG('A', 'y', 'B', 'R'):
        s->alt = 1;
    case MKTAG('A', 'y', 'b', 'R'):
        avctx->pix_fmt = AV_PIX_FMT_YUVA444P;
        s->decode_frame = decode_aybri;
        if (s->format != format) {
            ret  = build_vlc(&s->vlc[0], l_y_ybri, 256);
            ret |= build_vlc(&s->vlc[1], l_u_ybri, 256);
        break;
    case MKTAG(' ', 'Y', 'B', 'R'):
        s->alt = 1;
    case MKTAG(' ', 'Y', 'b', 'R'):
        avctx->pix_fmt = AV_PIX_FMT_YUV444P;
        s->decode_frame = decode_ybr;
        if (s->format != format) {
            ret  = build_vlc(&s->vlc[0], l_y_ybr, 256);
            ret |= build_vlc(&s->vlc[1], l_u_ybr, 256);
        break;
    case MKTAG(' ', 'y', 'B', 'R'):
        s->alt = 1;
    case MKTAG(' ', 'y', 'b', 'R'):
        avctx->pix_fmt = AV_PIX_FMT_YUV444P;
        s->decode_frame = decode_ybri;
        if (s->format != format) {
            ret  = build_vlc(&s->vlc[0], l_y_ybri, 256);
            ret |= build_vlc(&s->vlc[1], l_u_ybri, 256);
        break;
    case MKTAG('Y', 'B', 'R', 0x0a):
        avctx->pix_fmt = AV_PIX_FMT_YUV444P10;
        s->decode_frame = decode_ybr10;
        if (s->format != format) {
            ret  = build_vlc(&s->vlc[0], l_y_ybr10, 1024);
            ret |= build_vlc(&s->vlc[1], l_u_ybr10, 1024);
        break;
    case MKTAG('y', 'B', 'R', 0x0a):
        avctx->pix_fmt = AV_PIX_FMT_YUV444P10;
        s->decode_frame = decode_ybr10i;
        if (s->format != format) {
            ret  = build_vlc(&s->vlc[0], l_y_ybr10i, 1024);
            ret |= build_vlc(&s->vlc[1], l_u_ybr10i, 1024);
        break;
    case MKTAG('C', 'A', '4', 'p'):
        avctx->pix_fmt = AV_PIX_FMT_YUVA444P10;
        s->decode_frame = decode_ca4p;
        if (s->format != format) {
            ret  = build_vlc(&s->vlc[0], l_y_ybr10, 1024);
            ret |= build_vlc(&s->vlc[1], l_u_ybr10, 1024);
        break;
    case MKTAG('C', 'A', '4', 'i'):
        avctx->pix_fmt = AV_PIX_FMT_YUVA444P10;
        s->decode_frame = decode_ca4i;
        if (s->format != format) {
            ret  = build_vlc(&s->vlc[0], l_y_ybr10i, 1024);
            ret |= build_vlc(&s->vlc[1], l_u_ybr10i, 1024);
        break;
    case MKTAG('B', 'Y', 'R', 'Y'):
        avctx->pix_fmt = AV_PIX_FMT_YUV422P;
        s->decode_frame = decode_byry;
        if (s->format != format) {
            ret  = build_vlc(&s->vlc[0], l_y_byry, 256);
            ret |= build_vlc(&s->vlc[1], l_u_byry, 256);
        break;
    case MKTAG('B', 'Y', 'R', 'y'):
        avctx->pix_fmt = AV_PIX_FMT_YUV422P;
        s->decode_frame = decode_byryi;
        if (s->format != format) {
            ret  = build_vlc(&s->vlc[0], l_y_byryi, 256);
            ret |= build_vlc(&s->vlc[1], l_u_byryi, 256);
        break;
    case MKTAG('Y', 'b', 'Y', 'r'):
        avctx->pix_fmt = AV_PIX_FMT_YUV422P;
        s->decode_frame = decode_ybyr;
        if (s->format != format) {
            ret  = build_vlc(&s->vlc[0], l_y_ybyr, 256);
            ret |= build_vlc(&s->vlc[1], l_u_ybyr, 256);
        break;
    case MKTAG('C', '8', '2', 'p'):
        avctx->pix_fmt = AV_PIX_FMT_YUVA422P;
        s->decode_frame = decode_c82p;
        if (s->format != format) {
            ret  = build_vlc(&s->vlc[0], l_y_byry, 256);
            ret |= build_vlc(&s->vlc[1], l_u_byry, 256);
        break;
    case MKTAG('C', '8', '2', 'i'):
        avctx->pix_fmt = AV_PIX_FMT_YUVA422P;
        s->decode_frame = decode_c82i;
        if (s->format != format) {
            ret  = build_vlc(&s->vlc[0], l_y_byryi, 256);
            ret |= build_vlc(&s->vlc[1], l_u_byryi, 256);
        break;
    case MKTAG(0xa2, 'Y', 'R', 'Y'):
        avctx->pix_fmt = AV_PIX_FMT_YUV422P10;
        s->decode_frame = decode_yry10;
        if (s->format != format) {
            ret  = build_vlc(&s->vlc[0], l_y_yry10, 1024);
            ret |= build_vlc(&s->vlc[1], l_u_yry10, 1024);
        break;
    case MKTAG(0xa2, 'Y', 'R', 'y'):
        avctx->pix_fmt = AV_PIX_FMT_YUV422P10;
        s->decode_frame = decode_yry10i;
        if (s->format != format) {
            ret  = build_vlc(&s->vlc[0], l_y_yry10i, 1024);
            ret |= build_vlc(&s->vlc[1], l_u_yry10i, 1024);
        break;
    case MKTAG('C', 'A', '2', 'p'):
        avctx->pix_fmt = AV_PIX_FMT_YUVA422P10;
        s->decode_frame = decode_ca2p;
        if (s->format != format) {
            ret  = build_vlc(&s->vlc[0], l_y_yry10, 1024);
            ret |= build_vlc(&s->vlc[1], l_u_yry10, 1024);
        break;
    case MKTAG('C', 'A', '2', 'i'):
        avctx->pix_fmt = AV_PIX_FMT_YUVA422P10;
        s->decode_frame = decode_ca2i;
        if (s->format != format) {
            ret  = build_vlc(&s->vlc[0], l_y_yry10i, 1024);
            ret |= build_vlc(&s->vlc[1], l_u_yry10i, 1024);
        break;
    default:
        avpriv_request_sample(avctx, "unsupported format: 0x%X", format);
        return AVERROR_PATCHWELCOME;
    if (s->format != format) {
        if (ret < 0)
            return ret;
        s->format = format;
    p->pict_type = AV_PICTURE_TYPE_I;
    p->key_frame = 1;
    if ((ret = ff_thread_get_buffer(avctx, &frame, 0)) < 0)
        return ret;
    if ((ret = init_get_bits8(&gb, avpkt->data + 20, avpkt->size - 20)) < 0)
        return ret;
    s->decode_frame(avctx, p, &gb);
    *got_frame = 1;
    return avpkt->size;