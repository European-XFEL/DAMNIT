FROM debian:bookworm AS build

ENV PYTHONDONTWRITEBYTECODE=1

# install build dependencies
RUN set -eux; \
    apt update; \
    apt install -y --no-install-recommends \
        g++ \
        gcc \
        python3-dev \
        python3-pip \
        python3-venv \
        # include pyqt deps so pip does not install them later
        python3-pyqt5 \
        python3-pyqt5.qsci \
        python3-pyqt5.qtsvg \
    ; \
    apt list --installed $(apt-mark showmanual) > /.apt-deps-build

COPY . /src

# set up venv and python dependencies
RUN set -eux; \
    python3 -m venv --system-site-packages /app; \
    /app/bin/python3 -m pip install "/src[gui,backend]"; \
    /app/bin/python3 -m pip uninstall -y damnit; \
    /app/bin/python3 -m pip freeze > /.python-deps-build

FROM debian:bookworm-slim

COPY --from=build /app /app
COPY --from=build /.apt-deps-build /.apt-deps-build
COPY --from=build /.python-deps-build /.python-deps-build

ENV PATH="/app/bin:${PATH}"
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# install runtime dependencies
RUN set -eux; \
    apt update; \
    apt install -y --no-install-recommends \
        ca-certificates \
        python3-pyqt5 \
        python3-pyqt5.qsci \
        python3-pyqt5.qtsvg \
    ; \
    apt clean; \
    rm -rf /var/lib/apt/lists/*; \
    apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false; \
    apt list --installed $(apt-mark showmanual) > /.apt-deps-runtime

# install DAMNIT
COPY . /src
RUN set -eux; \
    /app/bin/python3 -m pip install -vv --no-cache-dir "/src[gui,backend]"; \
    /app/bin/python3 -m pip freeze > /.python-deps-runtime; \
    python3 -m pip check; \
    python3 -c "import damnit; print(damnit.__version__)"

CMD ["/app/bin/damnit", "gui"]
