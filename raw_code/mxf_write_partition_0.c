static void mxf_write_partition(AVFormatContext *s, int bodysid,

                                int indexsid,

                                const uint8_t *key, int write_metadata)

{

    MXFContext *mxf = s->priv_data;

    ByteIOContext *pb = s->pb;

    int64_t header_byte_count_offset;

    unsigned index_byte_count = 0;

    uint64_t partition_offset = url_ftell(pb);



    if (mxf->edit_units_count) {

        index_byte_count = 109 + (s->nb_streams+1)*6 +

            mxf->edit_units_count*(11+mxf->slice_count*4);

        // add encoded ber length

        index_byte_count += 16 + klv_ber_length(index_byte_count);

        index_byte_count += klv_fill_size(index_byte_count);

    }



    if (!memcmp(key, body_partition_key, 16)) {

        mxf->body_partition_offset =

            av_realloc(mxf->body_partition_offset,

                       (mxf->body_partitions_count+1)*

                       sizeof(*mxf->body_partition_offset));

        mxf->body_partition_offset[mxf->body_partitions_count++] = partition_offset;

    }



    // write klv

    put_buffer(pb, key, 16);



    klv_encode_ber_length(pb, 88 + 16 * mxf->essence_container_count);



    // write partition value

    put_be16(pb, 1); // majorVersion

    put_be16(pb, 2); // minorVersion

    put_be32(pb, KAG_SIZE); // KAGSize



    put_be64(pb, partition_offset); // ThisPartition



    if (!memcmp(key, body_partition_key, 16) && mxf->body_partitions_count > 1)

        put_be64(pb, mxf->body_partition_offset[mxf->body_partitions_count-2]); // PreviousPartition

    else if (!memcmp(key, footer_partition_key, 16))

        put_be64(pb, mxf->body_partition_offset[mxf->body_partitions_count-1]); // PreviousPartition

    else

        put_be64(pb, 0);



    put_be64(pb, mxf->footer_partition_offset); // footerPartition



    // set offset

    header_byte_count_offset = url_ftell(pb);

    put_be64(pb, 0); // headerByteCount, update later



    // indexTable

    put_be64(pb, index_byte_count); // indexByteCount

    put_be32(pb, index_byte_count ? indexsid : 0); // indexSID



    // BodyOffset

    if (bodysid && mxf->edit_units_count) {

        uint64_t partition_end = url_ftell(pb) + 8 + 4 + 16 + 8 +

            16*mxf->essence_container_count;

        put_be64(pb, partition_end + klv_fill_size(partition_end) +

                 index_byte_count - mxf->first_edit_unit_offset);

    } else

        put_be64(pb, 0);



    put_be32(pb, bodysid); // bodySID



    // operational pattern

    if (s->nb_streams > 1) {

        put_buffer(pb, op1a_ul, 14);

        put_be16(pb, 0x0900); // multi track

    } else {

        put_buffer(pb, op1a_ul, 16);

    }



    // essence container

    mxf_write_essence_container_refs(s);



    if (write_metadata) {

        // mark the start of the headermetadata and calculate metadata size

        int64_t pos, start;

        unsigned header_byte_count;



        mxf_write_klv_fill(s);

        start = url_ftell(s->pb);

        mxf_write_primer_pack(s);

        mxf_write_header_metadata_sets(s);

        pos = url_ftell(s->pb);

        header_byte_count = pos - start + klv_fill_size(pos);



        // update header_byte_count

        url_fseek(pb, header_byte_count_offset, SEEK_SET);

        put_be64(pb, header_byte_count);

        url_fseek(pb, pos, SEEK_SET);

    }



    put_flush_packet(pb);

}
