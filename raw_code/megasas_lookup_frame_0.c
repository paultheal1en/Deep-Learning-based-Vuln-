static MegasasCmd *megasas_lookup_frame(MegasasState *s,

    target_phys_addr_t frame)

{

    MegasasCmd *cmd = NULL;

    int num = 0, index;



    index = s->reply_queue_head;



    while (num < s->fw_cmds) {

        if (s->frames[index].pa && s->frames[index].pa == frame) {

            cmd = &s->frames[index];

            break;

        }

        index = megasas_next_index(s, index, s->fw_cmds);

        num++;

    }



    return cmd;

}
