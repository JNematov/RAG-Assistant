import { ComponentFixture, TestBed } from '@angular/core/testing';

import { PromptBubble } from './prompt-bubble';

describe('PromptBubble', () => {
  let component: PromptBubble;
  let fixture: ComponentFixture<PromptBubble>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [PromptBubble]
    })
    .compileComponents();

    fixture = TestBed.createComponent(PromptBubble);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
