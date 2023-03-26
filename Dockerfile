FROM mfisherman/openmpi

ENV APP_DIR "/app/"
ENV APP_SRC "${APP_DIR}main.cpp ${APP_DIR}matrix.h ${APP_DIR}utils.h"
ENV APP_EXC "${APP_DIR}main"

WORKDIR $APP_DIR

COPY ./ $APP_DIR

RUN mpic++ ${APP_SRC} -o ${APP_EXC}

#CMD mpirun -n 4 --oversubscribe ${APP_EXC} 128 128 128 0