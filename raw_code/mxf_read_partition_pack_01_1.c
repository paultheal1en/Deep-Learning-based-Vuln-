static int mxf_read_partition_pack(void *arg, AVIOContext *pb, int tag, int size, UID uid)

{

    MXFContext *mxf = arg;

    MXFPartition *partition;

    UID op;

    uint64_t footer_partition;



    if (mxf->partitions_count+1 >= UINT_MAX / sizeof(*mxf->partitions))

        return AVERROR(ENOMEM);



    mxf->partitions = av_realloc(mxf->partitions, (mxf->partitions_count + 1) * sizeof(*mxf->partitions));

    if (!mxf->partitions)

        return AVERROR(ENOMEM);



    if (mxf->parsing_backward) {

        /* insert the new partition pack in the middle

         * this makes the entries in mxf->partitions sorted by offset */

        memmove(&mxf->partitions[mxf->last_forward_partition+1],

                &mxf->partitions[mxf->last_forward_partition],

                (mxf->partitions_count - mxf->last_forward_partition)*sizeof(*mxf->partitions));

        partition = mxf->current_partition = &mxf->partitions[mxf->last_forward_partition];

    } else {

        mxf->last_forward_partition++;

        partition = mxf->current_partition = &mxf->partitions[mxf->partitions_count];

    }



    memset(partition, 0, sizeof(*partition));

    mxf->partitions_count++;



    switch(uid[13]) {

    case 2:

        partition->type = Header;

        break;

    case 3:

        partition->type = BodyPartition;

        break;

    case 4:

        partition->type = Footer;

        break;

    default:

        av_log(mxf->fc, AV_LOG_ERROR, "unknown partition type %i\n", uid[13]);

        return AVERROR_INVALIDDATA;

    }



    /* consider both footers to be closed (there is only Footer and CompleteFooter) */

    partition->closed = partition->type == Footer || !(uid[14] & 1);

    partition->complete = uid[14] > 2;

    avio_skip(pb, 8);

    partition->this_partition = avio_rb64(pb);

    partition->previous_partition = avio_rb64(pb);

    footer_partition = avio_rb64(pb);

    avio_skip(pb, 16);

    partition->index_sid = avio_rb32(pb);

    avio_skip(pb, 8);

    partition->body_sid = avio_rb32(pb);

    avio_read(pb, op, sizeof(UID));



    /* some files don'thave FooterPartition set in every partition */

    if (footer_partition) {

        if (mxf->footer_partition && mxf->footer_partition != footer_partition) {

            av_log(mxf->fc, AV_LOG_ERROR, "inconsistent FooterPartition value: %li != %li\n",

                   mxf->footer_partition, footer_partition);

        } else {

            mxf->footer_partition = footer_partition;

        }

    }



    av_dlog(mxf->fc, "PartitionPack: ThisPartition = 0x%lx, PreviousPartition = 0x%lx, "

            "FooterPartition = 0x%lx, IndexSID = %i, BodySID = %i\n",

            partition->this_partition,

            partition->previous_partition, footer_partition,

            partition->index_sid, partition->body_sid);



    if      (op[12] == 1 && op[13] == 1) mxf->op = OP1a;

    else if (op[12] == 1 && op[13] == 2) mxf->op = OP1b;

    else if (op[12] == 1 && op[13] == 3) mxf->op = OP1c;

    else if (op[12] == 2 && op[13] == 1) mxf->op = OP2a;

    else if (op[12] == 2 && op[13] == 2) mxf->op = OP2b;

    else if (op[12] == 2 && op[13] == 3) mxf->op = OP2c;

    else if (op[12] == 3 && op[13] == 1) mxf->op = OP3a;

    else if (op[12] == 3 && op[13] == 2) mxf->op = OP3b;

    else if (op[12] == 3 && op[13] == 3) mxf->op = OP3c;

    else if (op[12] == 0x10)             mxf->op = OPAtom;

    else

        av_log(mxf->fc, AV_LOG_ERROR, "unknown operational pattern: %02xh %02xh\n", op[12], op[13]);



    return 0;

}
