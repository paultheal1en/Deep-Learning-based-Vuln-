static int build_def_list(Picture *def, Picture **in, int len, int is_long, int sel)

{

    int  i[2] = { 0 };

    int index = 0;



    while (i[0] < len || i[1] < len) {

        while (i[0] < len && !(in[i[0]] && (in[i[0]]->reference & sel)))

            i[0]++;

        while (i[1] < len && !(in[i[1]] && (in[i[1]]->reference & (sel ^ 3))))

            i[1]++;

        if (i[0] < len) {

            in[i[0]]->pic_id = is_long ? i[0] : in[i[0]]->frame_num;

            split_field_copy(&def[index++], in[i[0]++], sel, 1);

        }

        if (i[1] < len) {

            in[i[1]]->pic_id = is_long ? i[1] : in[i[1]]->frame_num;

            split_field_copy(&def[index++], in[i[1]++], sel ^ 3, 0);

        }

    }



    return index;

}
