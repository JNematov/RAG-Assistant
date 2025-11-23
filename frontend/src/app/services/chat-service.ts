import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { Chat } from '../models/chat.model';

@Injectable({
  providedIn: 'root',
})
export class ChatService {
  constructor(private http: HttpClient) {}

  getChatList(): Observable<Chat[]> {
    return this.http.get<Chat[]>(`http://localhost:8000/health`);
  }
}
