void check_audio_video_inputs(int *has_video_ptr, int *has_audio_ptr)

{

    int has_video, has_audio, i, j;

    AVFormatContext *ic;



    has_video = 0;

    has_audio = 0;

    for(j=0;j<nb_input_files;j++) {

        ic = input_files[j];

        for(i=0;i<ic->nb_streams;i++) {

            AVCodecContext *enc = &ic->streams[i]->codec;

            switch(enc->codec_type) {

            case CODEC_TYPE_AUDIO:

                has_audio = 1;

                break;

            case CODEC_TYPE_VIDEO:

                has_video = 1;

                break;

            default:

                abort();

            }

        }

    }

    *has_video_ptr = has_video;

    *has_audio_ptr = has_audio;

}
