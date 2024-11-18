static int unpack_parse_unit(DiracParseUnit *pu, DiracParseContext *pc,

                             int offset)

{

    int8_t *start;



    if (offset < 0 || pc->index - 13 < offset)

        return 0;



    start = pc->buffer + offset;

    pu->pu_type = start[4];



    pu->next_pu_offset = AV_RB32(start + 5);

    pu->prev_pu_offset = AV_RB32(start + 9);



    if (pu->pu_type == 0x10 && pu->next_pu_offset == 0)

        pu->next_pu_offset = 13;



    if (pu->next_pu_offset && pu->next_pu_offset < 13) {

        av_log(NULL, AV_LOG_ERROR, "next_pu_offset %d is invalid\n", pu->next_pu_offset);

        return 0;

    }

    if (pu->prev_pu_offset && pu->prev_pu_offset < 13) {

        av_log(NULL, AV_LOG_ERROR, "prev_pu_offset %d is invalid\n", pu->prev_pu_offset);

        return 0;

    }



    return 1;

}
