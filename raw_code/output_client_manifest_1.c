static void output_client_manifest(struct VideoFiles *files,

                                   const char *basename, int split)

{

    char filename[1000];

    FILE *out;

    int i, j;



    if (split)

        snprintf(filename, sizeof(filename), "Manifest");

    else

        snprintf(filename, sizeof(filename), "%s.ismc", basename);

    out = fopen(filename, "w");

    if (!out) {

        perror(filename);

        return;

    }

    fprintf(out, "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n");

    fprintf(out, "<SmoothStreamingMedia MajorVersion=\"2\" MinorVersion=\"0\" "

                 "Duration=\"%"PRId64 "\">\n", files->duration * 10);

    if (files->video_file >= 0) {

        struct VideoFile *vf = files->files[files->video_file];

        int index = 0;

        fprintf(out,

                "\t<StreamIndex Type=\"video\" QualityLevels=\"%d\" "

                "Chunks=\"%d\" "

                "Url=\"QualityLevels({bitrate})/Fragments(video={start time})\">\n",

                files->nb_video_files, vf->chunks);

        for (i = 0; i < files->nb_files; i++) {

            vf = files->files[i];

            if (!vf->is_video)

                continue;

            fprintf(out,

                    "\t\t<QualityLevel Index=\"%d\" Bitrate=\"%d\" "

                    "FourCC=\"%s\" MaxWidth=\"%d\" MaxHeight=\"%d\" "

                    "CodecPrivateData=\"",

                    index, vf->bitrate, vf->fourcc, vf->width, vf->height);

            for (j = 0; j < vf->codec_private_size; j++)

                fprintf(out, "%02X", vf->codec_private[j]);

            fprintf(out, "\" />\n");

            index++;

        }

        vf = files->files[files->video_file];

        for (i = 0; i < vf->chunks; i++)

            fprintf(out, "\t\t<c n=\"%d\" d=\"%d\" />\n", i,

                    vf->offsets[i].duration);

        fprintf(out, "\t</StreamIndex>\n");

    }

    if (files->audio_file >= 0) {

        struct VideoFile *vf = files->files[files->audio_file];

        int index = 0;

        fprintf(out,

                "\t<StreamIndex Type=\"audio\" QualityLevels=\"%d\" "

                "Chunks=\"%d\" "

                "Url=\"QualityLevels({bitrate})/Fragments(audio={start time})\">\n",

                files->nb_audio_files, vf->chunks);

        for (i = 0; i < files->nb_files; i++) {

            vf = files->files[i];

            if (!vf->is_audio)

                continue;

            fprintf(out,

                    "\t\t<QualityLevel Index=\"%d\" Bitrate=\"%d\" "

                    "FourCC=\"%s\" SamplingRate=\"%d\" Channels=\"%d\" "

                    "BitsPerSample=\"16\" PacketSize=\"%d\" "

                    "AudioTag=\"%d\" CodecPrivateData=\"",

                    index, vf->bitrate, vf->fourcc, vf->sample_rate,

                    vf->channels, vf->blocksize, vf->tag);

            for (j = 0; j < vf->codec_private_size; j++)

                fprintf(out, "%02X", vf->codec_private[j]);

            fprintf(out, "\" />\n");

            index++;

        }

        vf = files->files[files->audio_file];

        for (i = 0; i < vf->chunks; i++)

            fprintf(out, "\t\t<c n=\"%d\" d=\"%d\" />\n",

                    i, vf->offsets[i].duration);

        fprintf(out, "\t</StreamIndex>\n");

    }

    fprintf(out, "</SmoothStreamingMedia>\n");

    fclose(out);

}
