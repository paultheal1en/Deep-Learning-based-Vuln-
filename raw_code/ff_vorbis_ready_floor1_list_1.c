void ff_vorbis_ready_floor1_list(vorbis_floor1_entry * list, int values)

{

    int i;

    list[0].sort = 0;

    list[1].sort = 1;

    for (i = 2; i < values; i++) {

        int j;

        list[i].low  = 0;

        list[i].high = 1;

        list[i].sort = i;

        for (j = 2; j < i; j++) {

            int tmp = list[j].x;

            if (tmp < list[i].x) {

                if (tmp > list[list[i].low].x)

                    list[i].low  =  j;

            } else {

                if (tmp < list[list[i].high].x)

                    list[i].high = j;

            }

        }

    }

    for (i = 0; i < values - 1; i++) {

        int j;

        for (j = i + 1; j < values; j++) {

            if (list[list[i].sort].x > list[list[j].sort].x) {

                int tmp = list[i].sort;

                list[i].sort = list[j].sort;

                list[j].sort = tmp;

            }

        }

    }

}
