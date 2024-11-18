static int mov_write_video_tag(ByteIOContext *pb, MOVTrack* track)

{

    int pos = url_ftell(pb);

    char compressor_name[32];

    int tag;



    put_be32(pb, 0); /* size */



    tag = track->enc->codec_tag;

    if (!tag)

    tag = codec_get_tag(codec_movvideo_tags, track->enc->codec_id);

    // if no mac fcc found, try with Microsoft tags

    if (!tag)

	tag = codec_get_tag(codec_bmp_tags, track->enc->codec_id);

    put_le32(pb, tag); // store it byteswapped



    put_be32(pb, 0); /* Reserved */

    put_be16(pb, 0); /* Reserved */

    put_be16(pb, 1); /* Data-reference index */



    put_be16(pb, 0); /* Codec stream version */

    put_be16(pb, 0); /* Codec stream revision (=0) */

    put_tag(pb, "FFMP"); /* Vendor */

    if(track->enc->codec_id == CODEC_ID_RAWVIDEO) {

        put_be32(pb, 0); /* Temporal Quality */

        put_be32(pb, 0x400); /* Spatial Quality = lossless*/

    } else {

        put_be32(pb, 0x200); /* Temporal Quality = normal */

        put_be32(pb, 0x200); /* Spatial Quality = normal */

    }

    put_be16(pb, track->enc->width); /* Video width */

    put_be16(pb, track->enc->height); /* Video height */

    put_be32(pb, 0x00480000); /* Horizontal resolution 72dpi */

    put_be32(pb, 0x00480000); /* Vertical resolution 72dpi */

    put_be32(pb, 0); /* Data size (= 0) */

    put_be16(pb, 1); /* Frame count (= 1) */

    

    memset(compressor_name,0,32);

    if (track->enc->codec->name)

        strncpy(compressor_name,track->enc->codec->name,31);

    put_byte(pb, FFMAX(strlen(compressor_name),32) );

    put_buffer(pb, compressor_name, 31);

    

    put_be16(pb, 0x18); /* Reserved */

    put_be16(pb, 0xffff); /* Reserved */

    if(track->enc->codec_id == CODEC_ID_MPEG4)

        mov_write_esds_tag(pb, track);

    else if(track->enc->codec_id == CODEC_ID_H263)

        mov_write_d263_tag(pb);

    else if(track->enc->codec_id == CODEC_ID_SVQ3)

        mov_write_svq3_tag(pb);    



    return updateSize (pb, pos);

}
