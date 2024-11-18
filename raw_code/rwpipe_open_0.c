static rwpipe *rwpipe_open( int argc, char *argv[] )

{

    rwpipe *this = av_mallocz( sizeof( rwpipe ) );



    if ( this != NULL )

    {

        int input[ 2 ];

        int output[ 2 ];



        pipe( input );

        pipe( output );



        this->pid = fork();



        if ( this->pid == 0 )

        {

#define COMMAND_SIZE 10240

            char *command = av_mallocz( COMMAND_SIZE );

            int i;



            strcpy( command, "" );

            for ( i = 0; i < argc; i ++ )

            {

                av_strlcat( command, argv[ i ], COMMAND_SIZE );

                av_strlcat( command, " ", COMMAND_SIZE );

            }



            dup2( output[ 0 ], STDIN_FILENO );

            dup2( input[ 1 ], STDOUT_FILENO );



            close( input[ 0 ] );

            close( input[ 1 ] );

            close( output[ 0 ] );

            close( output[ 1 ] );



            execl("/bin/sh", "sh", "-c", command, (char*)NULL );

            _exit( 255 );

        }

        else

        {

            close( input[ 1 ] );

            close( output[ 0 ] );



            this->reader = fdopen( input[ 0 ], "r" );

            this->writer = fdopen( output[ 1 ], "w" );

        }

    }



    return this;

}
