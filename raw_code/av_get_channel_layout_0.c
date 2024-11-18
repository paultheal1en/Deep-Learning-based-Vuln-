int64_t av_get_channel_layout(const char *name)

{

    int i = 0;

    do {

        if (!strcmp(channel_layout_map[i].name, name))

            return channel_layout_map[i].layout;

        i++;

    } while (channel_layout_map[i].name);



    return 0;

}
