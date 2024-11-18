static int analyze_chunk(AVFormatContext *s, const uint8_t *chunk)
{
    TYDemuxContext *ty = s->priv_data;
    int num_recs, i;
    TyRecHdr *hdrs;
    int num_6e0, num_be0, num_9c0, num_3c0;
    /* skip if it's a Part header */
    if (AV_RB32(&chunk[0]) == TIVO_PES_FILEID)
        return 0;
    /* number of records in chunk (we ignore high order byte;
     * rarely are there > 256 chunks & we don't need that many anyway) */
    num_recs = chunk[0];
    if (num_recs < 5) {
        /* try again with the next chunk.  Sometimes there are dead ones */
        return 0;
    }
    chunk += 4;       /* skip past rec count & SEQ bytes */
    ff_dlog(s, "probe: chunk has %d recs\n", num_recs);
    hdrs = parse_chunk_headers(chunk, num_recs);
    if (!hdrs)
        return AVERROR(ENOMEM);
    /* scan headers.
     * 1. check video packets.  Presence of 0x6e0 means S1.
     *    No 6e0 but have be0 means S2.
     * 2. probe for audio 0x9c0 vs 0x3c0 (AC3 vs Mpeg)
     *    If AC-3, then we have DTivo.
     *    If MPEG, search for PTS offset.  This will determine SA vs. DTivo.
     */
    num_6e0 = num_be0 = num_9c0 = num_3c0 = 0;
    for (i = 0; i < num_recs; i++) {
        switch (hdrs[i].subrec_type << 8 | hdrs[i].rec_type) {
        case 0x6e0:
            num_6e0++;
        case 0xbe0:
            num_be0++;
        case 0x3c0:
            num_3c0++;
        case 0x9c0:
            num_9c0++;
        }
    }
    ff_dlog(s, "probe: chunk has %d 0x6e0 recs, %d 0xbe0 recs.\n",
            num_6e0, num_be0);
    /* set up our variables */
    if (num_6e0 > 0) {
        ff_dlog(s, "detected Series 1 Tivo\n");
        ty->tivo_series = TIVO_SERIES1;
        ty->pes_length = SERIES1_PES_LENGTH;
    } else if (num_be0 > 0) {
        ff_dlog(s, "detected Series 2 Tivo\n");
        ty->tivo_series = TIVO_SERIES2;
        ty->pes_length = SERIES2_PES_LENGTH;
    }
    if (num_9c0 > 0) {
        ff_dlog(s, "detected AC-3 Audio (DTivo)\n");
        ty->audio_type = TIVO_AUDIO_AC3;
        ty->tivo_type = TIVO_TYPE_DTIVO;
        ty->pts_offset = AC3_PTS_OFFSET;
        ty->pes_length = AC3_PES_LENGTH;
    } else if (num_3c0 > 0) {
        ty->audio_type = TIVO_AUDIO_MPEG;
        ff_dlog(s, "detected MPEG Audio\n");
    }
    /* if tivo_type still unknown, we can check PTS location
     * in MPEG packets to determine tivo_type */
    if (ty->tivo_type == TIVO_TYPE_UNKNOWN) {
        uint32_t data_offset = 16 * num_recs;
        for (i = 0; i < num_recs; i++) {
            if ((hdrs[i].subrec_type << 0x08 | hdrs[i].rec_type) == 0x3c0 && hdrs[i].rec_size > 15) {
                /* first make sure we're aligned */
                int pes_offset = find_es_header(ty_MPEGAudioPacket,
                        &chunk[data_offset], 5);
                if (pes_offset >= 0) {
                    /* pes found. on SA, PES has hdr data at offset 6, not PTS. */
                    if ((chunk[data_offset + 6 + pes_offset] & 0x80) == 0x80) {
                        /* S1SA or S2(any) Mpeg Audio (PES hdr, not a PTS start) */
                        if (ty->tivo_series == TIVO_SERIES1)
                            ff_dlog(s, "detected Stand-Alone Tivo\n");
                        ty->tivo_type = TIVO_TYPE_SA;
                        ty->pts_offset = SA_PTS_OFFSET;
                    } else {
                        if (ty->tivo_series == TIVO_SERIES1)
                            ff_dlog(s, "detected DirecTV Tivo\n");
                        ty->tivo_type = TIVO_TYPE_DTIVO;
                        ty->pts_offset = DTIVO_PTS_OFFSET;
                    }
                }
            }
            data_offset += hdrs[i].rec_size;
        }
    }
    av_free(hdrs);
    return 0;
}