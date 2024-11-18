static void put_payload_header(

                                AVFormatContext *s,

                                ASFStream       *stream,

                                int             presentation_time,

                                int             m_obj_size,

                                int             m_obj_offset,

                                int             payload_len

            )

{

    ASFContext *asf = s->priv_data;

    ByteIOContext *pb = &asf->pb;

    int val;

    

    val = stream->num;

    if (s->streams[val - 1]->codec.coded_frame->key_frame)

        val |= ASF_PL_FLAG_KEY_FRAME;

    put_byte(pb, val);

        

    put_byte(pb, stream->seq);  //Media object number

    put_le32(pb, m_obj_offset); //Offset Into Media Object

         

    // Replicated Data shall be at least 8 bytes long.

    // The first 4 bytes of data shall contain the 

    // Size of the Media Object that the payload belongs to.

    // The next 4 bytes of data shall contain the 

    // Presentation Time for the media object that the payload belongs to.

    put_byte(pb, ASF_PAYLOAD_REPLICATED_DATA_LENGTH);



    put_le32(pb, m_obj_size);       //Replicated Data - Media Object Size

    put_le32(pb, presentation_time);//Replicated Data - Presentation Time

    

    if (asf->multi_payloads_present){

        put_le16(pb, payload_len);   //payload length

    }

}
