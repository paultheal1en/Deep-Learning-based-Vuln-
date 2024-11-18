static int imc_decode_block(AVCodecContext *avctx, IMCContext *q, int ch)
{
    int stream_format_code;
    int imc_hdr, i, j, ret;
    int flag;
    int bits, summer;
    int counter, bitscount;
    IMCChannel *chctx = q->chctx + ch;
    /* Check the frame header */
    imc_hdr = get_bits(&q->gb, 9);
    if (imc_hdr & 0x18) {
        av_log(avctx, AV_LOG_ERROR, "frame header check failed!\n");
        av_log(avctx, AV_LOG_ERROR, "got %X.\n", imc_hdr);
    stream_format_code = get_bits(&q->gb, 3);
    if (stream_format_code & 1) {
        av_log_ask_for_sample(avctx, "Stream format %X is not supported\n",
                              stream_format_code);
        return AVERROR_PATCHWELCOME;
    if (stream_format_code & 0x04)
        chctx->decoder_reset = 1;
    if (chctx->decoder_reset) {
        for (i = 0; i < BANDS; i++)
            chctx->old_floor[i] = 1.0;
        for (i = 0; i < COEFFS; i++)
            chctx->CWdecoded[i] = 0;
        chctx->decoder_reset = 0;
    flag = get_bits1(&q->gb);
    imc_read_level_coeffs(q, stream_format_code, chctx->levlCoeffBuf);
    if (stream_format_code & 0x4)
        imc_decode_level_coefficients(q, chctx->levlCoeffBuf,
                                      chctx->flcoeffs1, chctx->flcoeffs2);
    else
        imc_decode_level_coefficients2(q, chctx->levlCoeffBuf, chctx->old_floor,
                                       chctx->flcoeffs1, chctx->flcoeffs2);
    memcpy(chctx->old_floor, chctx->flcoeffs1, 32 * sizeof(float));
    counter = 0;
    for (i = 0; i < BANDS; i++) {
        if (chctx->levlCoeffBuf[i] == 16) {
            chctx->bandWidthT[i] = 0;
            counter++;
        } else
            chctx->bandWidthT[i] = band_tab[i + 1] - band_tab[i];
    memset(chctx->bandFlagsBuf, 0, BANDS * sizeof(int));
    for (i = 0; i < BANDS - 1; i++) {
        if (chctx->bandWidthT[i])
            chctx->bandFlagsBuf[i] = get_bits1(&q->gb);
    imc_calculate_coeffs(q, chctx->flcoeffs1, chctx->flcoeffs2, chctx->bandWidthT, chctx->flcoeffs3, chctx->flcoeffs5);
    bitscount = 0;
    /* first 4 bands will be assigned 5 bits per coefficient */
    if (stream_format_code & 0x2) {
        bitscount += 15;
        chctx->bitsBandT[0] = 5;
        chctx->CWlengthT[0] = 5;
        chctx->CWlengthT[1] = 5;
        chctx->CWlengthT[2] = 5;
        for (i = 1; i < 4; i++) {
            bits = (chctx->levlCoeffBuf[i] == 16) ? 0 : 5;
            chctx->bitsBandT[i] = bits;
            for (j = band_tab[i]; j < band_tab[i + 1]; j++) {
                chctx->CWlengthT[j] = bits;
                bitscount      += bits;
    if (avctx->codec_id == AV_CODEC_ID_IAC) {
        bitscount += !!chctx->bandWidthT[BANDS - 1];
        if (!(stream_format_code & 0x2))
            bitscount += 16;
    if ((ret = bit_allocation(q, chctx, stream_format_code,
                              512 - bitscount - get_bits_count(&q->gb),
                              flag)) < 0) {
        av_log(avctx, AV_LOG_ERROR, "Bit allocations failed\n");
        chctx->decoder_reset = 1;
        return ret;
    for (i = 0; i < BANDS; i++) {
        chctx->sumLenArr[i]   = 0;
        chctx->skipFlagRaw[i] = 0;
        for (j = band_tab[i]; j < band_tab[i + 1]; j++)
            chctx->sumLenArr[i] += chctx->CWlengthT[j];
        if (chctx->bandFlagsBuf[i])
            if ((((band_tab[i + 1] - band_tab[i]) * 1.5) > chctx->sumLenArr[i]) && (chctx->sumLenArr[i] > 0))
                chctx->skipFlagRaw[i] = 1;
    imc_get_skip_coeff(q, chctx);
    for (i = 0; i < BANDS; i++) {
        chctx->flcoeffs6[i] = chctx->flcoeffs1[i];
        /* band has flag set and at least one coded coefficient */
        if (chctx->bandFlagsBuf[i] && (band_tab[i + 1] - band_tab[i]) != chctx->skipFlagCount[i]) {
            chctx->flcoeffs6[i] *= q->sqrt_tab[ band_tab[i + 1] - band_tab[i]] /
                                   q->sqrt_tab[(band_tab[i + 1] - band_tab[i] - chctx->skipFlagCount[i])];
    /* calculate bits left, bits needed and adjust bit allocation */
    bits = summer = 0;
    for (i = 0; i < BANDS; i++) {
        if (chctx->bandFlagsBuf[i]) {
            for (j = band_tab[i]; j < band_tab[i + 1]; j++) {
                if (chctx->skipFlags[j]) {
                    summer += chctx->CWlengthT[j];
                    chctx->CWlengthT[j] = 0;
            bits   += chctx->skipFlagBits[i];
            summer -= chctx->skipFlagBits[i];
    imc_adjust_bit_allocation(q, chctx, summer);
    for (i = 0; i < BANDS; i++) {
        chctx->sumLenArr[i] = 0;
        for (j = band_tab[i]; j < band_tab[i + 1]; j++)
            if (!chctx->skipFlags[j])
                chctx->sumLenArr[i] += chctx->CWlengthT[j];
    memset(chctx->codewords, 0, sizeof(chctx->codewords));
    if (imc_get_coeffs(q, chctx) < 0) {
        av_log(avctx, AV_LOG_ERROR, "Read coefficients failed\n");
        chctx->decoder_reset = 1;
    if (inverse_quant_coeff(q, chctx, stream_format_code) < 0) {
        av_log(avctx, AV_LOG_ERROR, "Inverse quantization of coefficients failed\n");
        chctx->decoder_reset = 1;
    memset(chctx->skipFlags, 0, sizeof(chctx->skipFlags));
    imc_imdct256(q, chctx, avctx->channels);
    return 0;