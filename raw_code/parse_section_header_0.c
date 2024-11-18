static int parse_section_header(GetByteContext *gbc, int *section_size,

                                enum HapSectionType *section_type)

{

    if (bytestream2_get_bytes_left(gbc) < 4)

        return AVERROR_INVALIDDATA;



    *section_size = bytestream2_get_le24(gbc);

    *section_type = bytestream2_get_byte(gbc);



    if (*section_size == 0) {

        if (bytestream2_get_bytes_left(gbc) < 4)

            return AVERROR_INVALIDDATA;



        *section_size = bytestream2_get_le32(gbc);

    }



    if (*section_size > bytestream2_get_bytes_left(gbc))

        return AVERROR_INVALIDDATA;

    else

        return 0;

}
