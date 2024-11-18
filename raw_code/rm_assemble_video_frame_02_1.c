static int rm_assemble_video_frame(AVFormatContext *s, ByteIOContext *pb,

                                   RMDemuxContext *rm, RMStream *vst,

                                   AVPacket *pkt, int len)

{

    int hdr, seq, pic_num, len2, pos;

    int type;



    hdr = get_byte(pb); len--;

    type = hdr >> 6;



    if(type != 3){  // not frame as a part of packet

        seq = get_byte(pb); len--;

    }

    if(type != 1){  // not whole frame

        len2 = get_num(pb, &len);

        pos  = get_num(pb, &len);

        pic_num = get_byte(pb); len--;

    }

    if(len<0)

        return -1;

    rm->remaining_len = len;

    if(type&1){     // frame, not slice

        if(type == 3)  // frame as a part of packet

            len= len2;

        if(rm->remaining_len < len)

            return -1;

        rm->remaining_len -= len;

        if(av_new_packet(pkt, len + 9) < 0)

            return AVERROR(EIO);

        pkt->data[0] = 0;

        AV_WL32(pkt->data + 1, 1);

        AV_WL32(pkt->data + 5, 0);

        get_buffer(pb, pkt->data + 9, len);

        return 0;

    }

    //now we have to deal with single slice



    if((seq & 0x7F) == 1 || vst->curpic_num != pic_num){

        vst->slices = ((hdr & 0x3F) << 1) + 1;

        vst->videobufsize = len2 + 8*vst->slices + 1;

        av_free_packet(&vst->pkt); //FIXME this should be output.

        if(av_new_packet(&vst->pkt, vst->videobufsize) < 0)

            return AVERROR(ENOMEM);

        vst->videobufpos = 8*vst->slices + 1;

        vst->cur_slice = 0;

        vst->curpic_num = pic_num;

        vst->pktpos = url_ftell(pb);

    }

    if(type == 2)

        len = FFMIN(len, pos);



    if(++vst->cur_slice > vst->slices)

        return 1;

    AV_WL32(vst->pkt.data - 7 + 8*vst->cur_slice, 1);

    AV_WL32(vst->pkt.data - 3 + 8*vst->cur_slice, vst->videobufpos - 8*vst->slices - 1);

    if(vst->videobufpos + len > vst->videobufsize)

        return 1;

    if (get_buffer(pb, vst->pkt.data + vst->videobufpos, len) != len)

        return AVERROR(EIO);

    vst->videobufpos += len;

    rm->remaining_len-= len;



    if(type == 2 || (vst->videobufpos) == vst->videobufsize){

        vst->pkt.data[0] = vst->cur_slice-1;

        *pkt= vst->pkt;

        vst->pkt.data=

        vst->pkt.size= 0;

        if(vst->slices != vst->cur_slice) //FIXME find out how to set slices correct from the begin

            memmove(pkt->data + 1 + 8*vst->cur_slice, pkt->data + 1 + 8*vst->slices,

                vst->videobufpos - 1 - 8*vst->slices);

        pkt->size += 8*(vst->cur_slice - vst->slices);

        pkt->pts = AV_NOPTS_VALUE;

        pkt->pos = vst->pktpos;

        return 0;

    }



    return 1;

}
