static int add_crc_to_array(uint32_t crc, int64_t pts)

{

    if (size_of_array <= number_of_elements) {

        if (size_of_array == 0)

            size_of_array = 10;

        size_of_array *= 2;

        crc_array = av_realloc(crc_array, size_of_array * sizeof(uint32_t));

        pts_array = av_realloc(pts_array, size_of_array * sizeof(int64_t));

        if ((crc_array == NULL) || (pts_array == NULL)) {

            av_log(NULL, AV_LOG_ERROR, "Can't allocate array to store crcs\n");

            return AVERROR(ENOMEM);

        }

    }

    crc_array[number_of_elements] = crc;

    pts_array[number_of_elements] = pts;

    number_of_elements++;

    return 0;

}
