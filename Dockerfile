#######################################################################################################################
# Build with this command:                                                                                            #
#       docker build -t [Barry121/graph-neural-network|Barry121/graph-neural-network:<#version>] .                    #
#######################################################################################################################

# Start from ucsdets/scipy-ml-notebook
FROM ucsdets/scipy-ml-notebook

# Add everything to project folder
ADD . /dsc180-graph-neural-net

# OR Clone the Git Repo
# ARG username=${Barry0121}
# ARG password=${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}

# Install packages not in the original image
RUN conda create --name gcn --file ./environment.txt

# Activate the environment
RUN conda activate gcn