static void init_demo(const char *filename)

{

    int i, j;

    int h;

    int radian;

    char line[3 * W];



    FILE *fichier;



    fichier = fopen(filename, "rb");

    if (!fichier) {

        perror(filename);

        exit(1);

    }



    fread(line, 1, 15, fichier);

    for (i = 0; i < H; i++) {

        fread(line, 1, 3 * W, fichier);

        for (j = 0; j < W; j++) {

            tab_r[W * i + j] = line[3 * j    ];

            tab_g[W * i + j] = line[3 * j + 1];

            tab_b[W * i + j] = line[3 * j + 2];

        }

    }

    fclose(fichier);



    /* tables sin/cos */

    for (i = 0; i < 360; i++) {

        radian = 2 * i * MY_PI / 360;

        h      = 2 * FIXP + int_sin (radian);

        h_cos[i] = h * int_sin(radian + MY_PI / 2) / 2 / FIXP;

        h_sin[i] = h * int_sin(radian)             / 2 / FIXP;

    }

}
