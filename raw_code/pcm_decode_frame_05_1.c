static int pcm_decode_frame(AVCodecContext *avctx, void *data,
                            int *got_frame_ptr, AVPacket *avpkt)
{
    const uint8_t *src = avpkt->data;
    int buf_size       = avpkt->size;
    PCMDecode *s       = avctx->priv_data;
    int sample_size, c, n, ret, samples_per_block;
    uint8_t *samples;
    int32_t *dst_int32_t;
    sample_size = av_get_bits_per_sample(avctx->codec_id) / 8;
    /* av_get_bits_per_sample returns 0 for AV_CODEC_ID_PCM_DVD */
    samples_per_block = 1;
    if (AV_CODEC_ID_PCM_DVD == avctx->codec_id) {
        if (avctx->bits_per_coded_sample != 20 &&
            avctx->bits_per_coded_sample != 24) {
            av_log(avctx, AV_LOG_ERROR,
                   "PCM DVD unsupported sample depth %i\n",
                   avctx->bits_per_coded_sample);
        /* 2 samples are interleaved per block in PCM_DVD */
        samples_per_block = 2;
        sample_size       = avctx->bits_per_coded_sample * 2 / 8;
    } else if (avctx->codec_id == AV_CODEC_ID_PCM_LXF) {
        /* we process 40-bit blocks per channel for LXF */
        samples_per_block = 2;
        sample_size       = 5;
    if (sample_size == 0) {
        av_log(avctx, AV_LOG_ERROR, "Invalid sample_size\n");
    n = avctx->channels * sample_size;
    if (n && buf_size % n) {
        if (buf_size < n) {
            av_log(avctx, AV_LOG_ERROR,
                   "Invalid PCM packet, data has size %d but at least a size of %d was expected\n",
                   buf_size, n);
            return AVERROR_INVALIDDATA;
        } else
            buf_size -= buf_size % n;
    n = buf_size / sample_size;
    /* get output buffer */
    s->frame.nb_samples = n * samples_per_block / avctx->channels;
    if ((ret = avctx->get_buffer(avctx, &s->frame)) < 0) {
        av_log(avctx, AV_LOG_ERROR, "get_buffer() failed\n");
        return ret;
    samples = s->frame.data[0];
    switch (avctx->codec->id) {
    case AV_CODEC_ID_PCM_U32LE:
        DECODE(32, le32, src, samples, n, 0, 0x80000000)
        break;
    case AV_CODEC_ID_PCM_U32BE:
        DECODE(32, be32, src, samples, n, 0, 0x80000000)
        break;
    case AV_CODEC_ID_PCM_S24LE:
        DECODE(32, le24, src, samples, n, 8, 0)
        break;
    case AV_CODEC_ID_PCM_S24BE:
        DECODE(32, be24, src, samples, n, 8, 0)
        break;
    case AV_CODEC_ID_PCM_U24LE:
        DECODE(32, le24, src, samples, n, 8, 0x800000)
        break;
    case AV_CODEC_ID_PCM_U24BE:
        DECODE(32, be24, src, samples, n, 8, 0x800000)
        break;
    case AV_CODEC_ID_PCM_S24DAUD:
        for (; n > 0; n--) {
            uint32_t v = bytestream_get_be24(&src);
            v >>= 4; // sync flags are here
            AV_WN16A(samples, ff_reverse[(v >> 8) & 0xff] +
                             (ff_reverse[v        & 0xff] << 8));
            samples += 2;
        break;
    case AV_CODEC_ID_PCM_S16LE_PLANAR:
    {
        int i;
        n /= avctx->channels;
        for (c = 0; c < avctx->channels; c++) {
            samples = s->frame.extended_data[c];
            for (i = n; i > 0; i--) {
                AV_WN16A(samples, bytestream_get_le16(&src));
                samples += 2;
        break;
    case AV_CODEC_ID_PCM_U16LE:
        DECODE(16, le16, src, samples, n, 0, 0x8000)
        break;
    case AV_CODEC_ID_PCM_U16BE:
        DECODE(16, be16, src, samples, n, 0, 0x8000)
        break;
    case AV_CODEC_ID_PCM_S8:
        for (; n > 0; n--)
            *samples++ = *src++ + 128;
        break;
#if HAVE_BIGENDIAN
    case AV_CODEC_ID_PCM_F64LE:
        DECODE(64, le64, src, samples, n, 0, 0)
        break;
    case AV_CODEC_ID_PCM_S32LE:
    case AV_CODEC_ID_PCM_F32LE:
        DECODE(32, le32, src, samples, n, 0, 0)
        break;
    case AV_CODEC_ID_PCM_S16LE:
        DECODE(16, le16, src, samples, n, 0, 0)
        break;
    case AV_CODEC_ID_PCM_F64BE:
    case AV_CODEC_ID_PCM_F32BE:
    case AV_CODEC_ID_PCM_S32BE:
    case AV_CODEC_ID_PCM_S16BE:
#else
    case AV_CODEC_ID_PCM_F64BE:
        DECODE(64, be64, src, samples, n, 0, 0)
        break;
    case AV_CODEC_ID_PCM_F32BE:
    case AV_CODEC_ID_PCM_S32BE:
        DECODE(32, be32, src, samples, n, 0, 0)
        break;
    case AV_CODEC_ID_PCM_S16BE:
        DECODE(16, be16, src, samples, n, 0, 0)
        break;
    case AV_CODEC_ID_PCM_F64LE:
    case AV_CODEC_ID_PCM_F32LE:
    case AV_CODEC_ID_PCM_S32LE:
    case AV_CODEC_ID_PCM_S16LE:
#endif /* HAVE_BIGENDIAN */
    case AV_CODEC_ID_PCM_U8:
        memcpy(samples, src, n * sample_size);
        break;
    case AV_CODEC_ID_PCM_ZORK:
        for (; n > 0; n--) {
            int v = *src++;
            if (v < 128)
                v = 128 - v;
            *samples++ = v;
        break;
    case AV_CODEC_ID_PCM_ALAW:
    case AV_CODEC_ID_PCM_MULAW:
        for (; n > 0; n--) {
            AV_WN16A(samples, s->table[*src++]);
            samples += 2;
        break;
    case AV_CODEC_ID_PCM_DVD:
    {
        const uint8_t *src8;
        dst_int32_t = (int32_t *)s->frame.data[0];
        n /= avctx->channels;
        switch (avctx->bits_per_coded_sample) {
        case 20:
            while (n--) {
                c    = avctx->channels;
                src8 = src + 4 * c;
                while (c--) {
                    *dst_int32_t++ = (bytestream_get_be16(&src) << 16) + ((*src8   & 0xf0) <<  8);
                    *dst_int32_t++ = (bytestream_get_be16(&src) << 16) + ((*src8++ & 0x0f) << 12);
                src = src8;
            break;
        case 24:
            while (n--) {
                c    = avctx->channels;
                src8 = src + 4 * c;
                while (c--) {
                    *dst_int32_t++ = (bytestream_get_be16(&src) << 16) + ((*src8++) << 8);
                    *dst_int32_t++ = (bytestream_get_be16(&src) << 16) + ((*src8++) << 8);
                src = src8;
            break;
        break;
    case AV_CODEC_ID_PCM_LXF:
    {
        int i;
        n /= avctx->channels;
        for (c = 0; c < avctx->channels; c++) {
            dst_int32_t = (int32_t *)s->frame.extended_data[c];
            for (i = 0; i < n; i++) {
                // extract low 20 bits and expand to 32 bits
                *dst_int32_t++ =  (src[2]         << 28) |
                                  (src[1]         << 20) |
                                  (src[0]         << 12) |
                                 ((src[2] & 0x0F) <<  8) |
                                   src[1];
                // extract high 20 bits and expand to 32 bits
                *dst_int32_t++ =  (src[4]         << 24) |
                                  (src[3]         << 16) |
                                 ((src[2] & 0xF0) <<  8) |
                                  (src[4]         <<  4) |
                                  (src[3]         >>  4);
                src += 5;
        break;
    default:
        return -1;
    *got_frame_ptr   = 1;
    *(AVFrame *)data = s->frame;
    return buf_size;