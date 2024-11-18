static void dump_human_image_check(ImageCheck *check)

{

    if (!(check->corruptions || check->leaks || check->check_errors)) {

        printf("No errors were found on the image.\n");

    } else {

        if (check->corruptions) {

            printf("\n%" PRId64 " errors were found on the image.\n"

                "Data may be corrupted, or further writes to the image "

                "may corrupt it.\n",

                check->corruptions);

        }



        if (check->leaks) {

            printf("\n%" PRId64 " leaked clusters were found on the image.\n"

                "This means waste of disk space, but no harm to data.\n",

                check->leaks);

        }



        if (check->check_errors) {

            printf("\n%" PRId64 " internal errors have occurred during the check.\n",

                check->check_errors);

        }

    }



    if (check->total_clusters != 0 && check->allocated_clusters != 0) {

        printf("%" PRId64 "/%" PRId64 "= %0.2f%% allocated, %0.2f%% fragmented\n",

        check->allocated_clusters, check->total_clusters,

        check->allocated_clusters * 100.0 / check->total_clusters,

        check->fragmented_clusters * 100.0 / check->allocated_clusters);

    }



    if (check->image_end_offset) {

        printf("Image end offset: %" PRId64 "\n", check->image_end_offset);

    }

}
