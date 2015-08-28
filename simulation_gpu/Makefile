CC  = g++
CFLAGS 	= -g -std=c++0x -O4

ifeq ("$(shell uname)", "Darwin")
  CC = g++
  CFLAGS += -Wno-deprecated-declarations
  LDFLAGS = -framework Foundation -framework GLUT -framework OpenGL -lm -lOpenImageIO -ltbb
else
  ifeq ("$(shell uname)", "Linux")
  	CFLAGS += -I/usr/local/cuda/include 
    LDFLAGS = -L/usr/lib64/ -L/group/dpa/lib -lX11 -lGL -lGLU -lglut -lGLEW -lm -lOpenCL
  endif
endif


OBJECTS = ${addsuffix .o, ${basename ${shell find *.cpp}}}
DEPS = $(OBJECTS:.o=.d)

PROJECT = main

.PHONY: echo clean clang setclang

g++: | setg++ ${PROJECT}

setg++:
	$(eval CXX = g++)
	$(eval CXX_LINK = g++)

echo:
	@echo CC $(CC)
	@echo CFLAGS $(CFLAGS)
	@echo LDFLAGS $(LDFLAGS)
	@echo OBJECTS $(OBJECTS)
	@echo DEPS $(DEPS)
	@echo PROJECT $(PROJECT)


debug: DEPS $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

${OBJECTS}: %.o: %.cpp
	${CC} ${CFLAGS} -c $< -o $@

${DEPS}: %.d: %.cpp
	@set -e; rm -f $@; \
 	$(CC) -MM $(CFLAGS) $< > $@.$$$$; \
 	sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@; \
 	rm -f $@.$$$$

${PROJECT}:	${OBJECTS}
	${CC} $(LDFLAGS) -o $@ ${OBJECTS} ${LDFLAGS}

clean:
	rm -rf core.* ${OBJECTS} ${DEPS} ${PROJECT} *~ *.d.* *conflict*;

-include $(DEPS)
