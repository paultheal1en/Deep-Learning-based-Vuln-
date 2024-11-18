static int prepare_sdp_description(FFStream *stream, uint8_t **pbuffer,

                                   struct in_addr my_ip)

{

    AVFormatContext *avc;

    AVStream avs[MAX_STREAMS];

    int i;



    avc =  avformat_alloc_context();

    if (avc == NULL) {

        return -1;

    }

    av_metadata_set2(&avc->metadata, "title",

                     stream->title[0] ? stream->title : "No Title", 0);

    avc->nb_streams = stream->nb_streams;

    if (stream->is_multicast) {

        snprintf(avc->filename, 1024, "rtp://%s:%d?multicast=1?ttl=%d",

                 inet_ntoa(stream->multicast_ip),

                 stream->multicast_port, stream->multicast_ttl);

    } else {

        snprintf(avc->filename, 1024, "rtp://0.0.0.0");

    }



    for(i = 0; i < stream->nb_streams; i++) {

        avc->streams[i] = &avs[i];

        avc->streams[i]->codec = stream->streams[i]->codec;

    }

    *pbuffer = av_mallocz(2048);

    avf_sdp_create(&avc, 1, *pbuffer, 2048);


    av_free(avc);



    return strlen(*pbuffer);

}