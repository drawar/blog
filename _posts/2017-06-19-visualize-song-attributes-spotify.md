---
layout: post
title:  "Visualizing song attributes from Spotify"
date:   2017-06-19
tags: spotify
image: /assets/article_images/2017-06-19-visualize-song-attributes-spotify/spotify-top-song.jpg
comments: true
---

As a die-hard music fan and a long time Spotify user, I was excited to have discovered Spotify Web API a couple of days ago. I stumbled upon an analysis of Radiohead's gloomiest songs using a combination of audio features scraped from Spotify API and that gave me some great idea to do a statistical analysis of my favorite songs.

There's hardly a day gone by that I haven't used Spotify for at least half an hour, and that gave Spotify enough information to curate a playlist of my most streamed songs of 2016. The playlist contains 100 songs covering a wide spectrum of genres, and I was quite curious what insights I could learn about my musical preferences.
## Connecting to Spotify Web API
<br>
To get started with Spotify Web API, you have to first set up a developer account [here](https://developer.spotify.com/my-applications/#!/applications), register an application, and get your *Client ID* and *Client Secret*, both of which are needed to generate your access token and header value for your HTTP GET request.

{% highlight r %}
library(tidyverse)
library(httr)


clientID = 'xxxxxxxxxxxxxx'
clientSecret = 'xxxxxxxxxxxxxx'

# Get header value from Client ID and Client Secret
# Dependencies: tidyverse, httr
get_header_value <- function(client_id, client_secret) {
  token <- POST(
    'https://accounts.spotify.com/api/token',
    accept_json(),
    authenticate(clientID, secret),
    body = list(grant_type = 'client_credentials'),
    encode = 'form',
    verbose()
  ) %>% content %>% .$access_token
  header_value <- paste0('Bearer ', token)
  return(header_value)
}

header_value <- get_header_value(clientID, clientSecret)

{% endhighlight %}
## Getting attributes from songs
<br>
Next, I grabbed the track info for my *Top Songs of 2016* playlist and preprocessed it into a nice tibble (a modern dataframe)

{% highlight r %}

# Get all tracks from Playlist ID and User ID
# Dependencies: tidyverse, httr
get_playlist_tracks <- function(user_id, playlist_id, header_value) {
    tracks <- GET(paste0('https://api.spotify.com/v1/users/', user_id,'/playlists/', playlist_id, '/tracks'),
                  add_headers(Authorization = header_value)) %>% 
      content %>% 
      .$items 
    
    track_names <- sapply(1:length(tracks), function(t) tracks[[t]]$track$name)
    
    ids <- map(1:length(tracks), function(z) tracks[[z]]$track$id) %>% 
      unlist %>% paste0(collapse=',')
    
    res <- GET(paste0('https://api.spotify.com/v1/audio-features/?ids=', ids),
               add_headers(Authorization = header_value)) %>% content %>% .$audio_features
    
    df <- unlist(res) %>% 
      matrix(nrow = length(res), byrow = T) %>% 
      as.data.frame(stringsAsFactors = F)
    names(df) <- names(res[[1]])
    df['track_name'] <- track_names
    df <- as_tibble(df) %>% 
    # convert from character to numeric
    mutate_at(c('danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness',
                  'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature'), funs(as.numeric(gsub('[^0-9.-e]+', '', as.character(.))))) 
    return(df)
}

user_id <- 'spotify'
playlist_id <- '37i9dQZF1CyXyPUy57ueoa'

top_song <- get_playlist_tracks(user_id, playlist_id, header_value)
top_song <- top_song %>% select(-c(mode, instrumentalness, liveness, time_signature, uri, track_href, analysis_url, type))
str(top_song)
{% endhighlight %}

I decided to exclude some attributes that I deemed not relevant to this analysis, namely **mode**, **instrumentalness**, **liveness**, **time signature**, etc. 

{% highlight r %}
## Classes 'tbl_df', 'tbl' and 'data.frame':    100 obs. of  11 variables:
##  $ danceability: num  0.617 0.399 0.58 0.422 0.842 0.636 0.447 0.803 0.634 0.677 ...
##  $ energy      : num  0.526 0.787 0.922 0.712 0.476 0.741 0.694 0.631 0.577 0.604 ...
##  $ key         : num  6 1 1 11 0 11 7 8 4 7 ...
##  $ loudness    : num  7.9 2.88 3.79 5.91 7.92 ...
##  $ speechiness : num  0.0519 0.0499 0.0958 0.1 0.0626 0.041 0.382 0.0444 0.0303 0.0385 ...
##  $ acousticness: num  0.405 0.0197 0.0917 0.273 0.0263 0.0132 0.0293 0.102 0.143 0.0612 ...
##  $ valence     : num  0.574 0.574 0.676 0.451 0.768 0.546 0.503 0.659 0.387 0.299 ...
##  $ tempo       : num  82.1 117.1 89.8 78.5 140 ...
##  $ id          : chr  "2A4nONaOJXSDBvezgxyAV4" "4VrWlk8IQxevMvERoX08iC" "2x5qF66rFO6DERBMNkQAqn" "2CvOqDpQIMw69cCzWqr5yr" ...
##  $ duration_ms : num  215427 216120 200387 261160 201400 ...
##  $ track_name  : chr  "Take A Bow (Glee Cast Version)" "Chandelier" "(There's Gotta Be) More To Life" "Halo" ...
{% endhighlight %}

The attributes, or *features* in machine learning terminology, are the variables that our mathematical models take into consideration when they try to predict stuff. Our final tibble contain the following attributes for each song:
- **Tempo**: The tempo of the song, measured by beats per minute (BPM).
- **Energy**:The energy of a song, the higher the value, the more energetic.
- **Danceability**: The higher the value, the easier it is to dance to this song.
- **Loudness**: The higher the value, the louder the song (in dB).
- **Valence**: The higher the value, the more positive mood for the song.
- **Length**: The duration of the song, in ms.
- **Acousticness**: The higher the value the more acoustic the song is.
- **Key**: The higher the value the pitchier the song is.
- **Speechiness**: The higher the value the more speech-like the song is i.e. consisted of mainly spoken words

