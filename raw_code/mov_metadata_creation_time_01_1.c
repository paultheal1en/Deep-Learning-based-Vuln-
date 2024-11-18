static void mov_metadata_creation_time(AVMetadata **metadata, time_t time)

{

    char buffer[32];

    if (time) {

        time -= 2082844800;  /* seconds between 1904-01-01 and Epoch */

        strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", gmtime(&time));

        av_metadata_set2(metadata, "creation_time", buffer, 0);

    }

}
