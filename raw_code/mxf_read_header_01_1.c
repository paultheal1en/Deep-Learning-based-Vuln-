static int mxf_read_header(AVFormatContext *s)

{

    MXFContext *mxf = s->priv_data;

    KLVPacket klv;

    int64_t essence_offset = 0;

    int ret;



    mxf->last_forward_tell = INT64_MAX;

    mxf->edit_units_per_packet = 1;



    if (!mxf_read_sync(s->pb, mxf_header_partition_pack_key, 14)) {

        av_log(s, AV_LOG_ERROR, "could not find header partition pack key\n");

        return AVERROR_INVALIDDATA;

    }

    avio_seek(s->pb, -14, SEEK_CUR);

    mxf->fc = s;

    mxf->run_in = avio_tell(s->pb);



    while (!url_feof(s->pb)) {

        const MXFMetadataReadTableEntry *metadata;



        if (klv_read_packet(&klv, s->pb) < 0) {

            /* EOF - seek to previous partition or stop */

            if(mxf_parse_handle_partition_or_eof(mxf) <= 0)

                break;

            else

                continue;

        }



        PRINT_KEY(s, "read header", klv.key);

        av_dlog(s, "size %"PRIu64" offset %#"PRIx64"\n", klv.length, klv.offset);

        if (IS_KLV_KEY(klv.key, mxf_encrypted_triplet_key) ||

            IS_KLV_KEY(klv.key, mxf_essence_element_key) ||

            IS_KLV_KEY(klv.key, mxf_avid_essence_element_key) ||

            IS_KLV_KEY(klv.key, mxf_system_item_key)) {



            if (!mxf->current_partition) {

                av_log(mxf->fc, AV_LOG_ERROR, "found essence prior to first PartitionPack\n");

                return AVERROR_INVALIDDATA;

            }



            if (!mxf->current_partition->essence_offset) {

                /* for OP1a we compute essence_offset

                 * for OPAtom we point essence_offset after the KL (usually op1a_essence_offset + 20 or 25)

                 * TODO: for OP1a we could eliminate this entire if statement, always stopping parsing at op1a_essence_offset

                 *       for OPAtom we still need the actual essence_offset though (the KL's length can vary)

                 */

                int64_t op1a_essence_offset =

                    round_to_kag(mxf->current_partition->this_partition +

                                 mxf->current_partition->pack_length,       mxf->current_partition->kag_size) +

                    round_to_kag(mxf->current_partition->header_byte_count, mxf->current_partition->kag_size) +

                    round_to_kag(mxf->current_partition->index_byte_count,  mxf->current_partition->kag_size);



                if (mxf->op == OPAtom) {

                    /* point essence_offset to the actual data

                    * OPAtom has all the essence in one big KLV

                    */

                    mxf->current_partition->essence_offset = avio_tell(s->pb);

                    mxf->current_partition->essence_length = klv.length;

                } else {

                    /* NOTE: op1a_essence_offset may be less than to klv.offset (C0023S01.mxf)  */

                    mxf->current_partition->essence_offset = op1a_essence_offset;

                }

            }



            if (!essence_offset)

                essence_offset = klv.offset;



            /* seek to footer, previous partition or stop */

            if (mxf_parse_handle_essence(mxf) <= 0)

                break;

            continue;

        } else if (!memcmp(klv.key, mxf_header_partition_pack_key, 13) &&

                   klv.key[13] >= 2 && klv.key[13] <= 4 && mxf->current_partition) {

            /* next partition pack - keep going, seek to previous partition or stop */

            if(mxf_parse_handle_partition_or_eof(mxf) <= 0)

                break;

            else if (mxf->parsing_backward)

                continue;

            /* we're still parsing forward. proceed to parsing this partition pack */

        }



        for (metadata = mxf_metadata_read_table; metadata->read; metadata++) {

            if (IS_KLV_KEY(klv.key, metadata->key)) {

                int res;

                if (klv.key[5] == 0x53) {

                    res = mxf_read_local_tags(mxf, &klv, metadata->read, metadata->ctx_size, metadata->type);

                } else {

                    uint64_t next = avio_tell(s->pb) + klv.length;

                    res = metadata->read(mxf, s->pb, 0, klv.length, klv.key, klv.offset);



                    /* only seek forward, else this can loop for a long time */

                    if (avio_tell(s->pb) > next) {

                        av_log(s, AV_LOG_ERROR, "read past end of KLV @ %#"PRIx64"\n",

                               klv.offset);

                        return AVERROR_INVALIDDATA;

                    }



                    avio_seek(s->pb, next, SEEK_SET);

                }

                if (res < 0) {

                    av_log(s, AV_LOG_ERROR, "error reading header metadata\n");

                    return res;

                }

                break;

            }

        }

        if (!metadata->read)

            avio_skip(s->pb, klv.length);

    }

    /* FIXME avoid seek */

    if (!essence_offset)  {

        av_log(s, AV_LOG_ERROR, "no essence\n");

        return AVERROR_INVALIDDATA;

    }

    avio_seek(s->pb, essence_offset, SEEK_SET);



    mxf_compute_essence_containers(mxf);



    /* we need to do this before computing the index tables

     * to be able to fill in zero IndexDurations with st->duration */

    if ((ret = mxf_parse_structural_metadata(mxf)) < 0)

        return ret;



    if ((ret = mxf_compute_index_tables(mxf)) < 0)

        return ret;



    if (mxf->nb_index_tables > 1) {

        /* TODO: look up which IndexSID to use via EssenceContainerData */

        av_log(mxf->fc, AV_LOG_INFO, "got %i index tables - only the first one (IndexSID %i) will be used\n",

               mxf->nb_index_tables, mxf->index_tables[0].index_sid);

    } else if (mxf->nb_index_tables == 0 && mxf->op == OPAtom) {

        av_log(mxf->fc, AV_LOG_ERROR, "cannot demux OPAtom without an index\n");

        return AVERROR_INVALIDDATA;

    }



    mxf_handle_small_eubc(s);



    return 0;

}
